from datetime import timedelta
import pandas as pd
from joblib import dump

from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier

from alpaca_data import AlpacaData, safe_end_for_free_feed
from config import SYMBOLS, SPY_SYMBOL, MODEL_PATH
from feature_engineering import add_features, add_spy_features, add_cross_sectional_features


HORIZON_BARS = 1     # predict the direction of the next 15-min bar
HISTORY_DAYS = 120
TEST_FRACTION = 0.2  # last 20% of the timeline is held out (and later used by the backtest)


def make_labeled_symbol_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = add_features(df)

    df["future_close"] = df["close"].shift(-HORIZON_BARS)
    df["future_ret"] = (df["future_close"] / df["close"]) - 1.0
    df["y"] = (df["future_ret"] > 0).astype(int)

    df = df.dropna()
    return df


def main():
    data = AlpacaData()

    end = safe_end_for_free_feed()
    start = end - timedelta(days=HISTORY_DAYS)

    # 1) Download + label each symbol
    labeled_list = []
    for sym in SYMBOLS:
        print(f"Downloading {sym} ...")
        raw = data.get_bars(sym, start_utc=start, end_utc=end)
        labeled_list.append(make_labeled_symbol_df(raw))

    ds = pd.concat(labeled_list, ignore_index=True)

    # 2) Merge SPY features by timestamp (cross-asset regime)
    spy_raw = data.get_bars(SPY_SYMBOL, start_utc=start, end_utc=end)
    ds = add_spy_features(ds, spy_raw)

    # 3) Universe cross-sectional features (mean/std of ret_1 across symbols at each timestamp)
    ds = add_cross_sectional_features(ds)

    # 4) One-hot encode symbol
    ds = pd.get_dummies(ds, columns=["symbol"], drop_first=False)

    # 5) Train/test split by time (all symbols at a given timestamp land on the same side)
    ds = ds.sort_values("timestamp").reset_index(drop=True)

    drop_cols = {"timestamp", "future_close", "future_ret", "y"}
    feature_cols = [c for c in ds.columns if c not in drop_cols]

    test_start = ds["timestamp"].iloc[int(len(ds) * (1 - TEST_FRACTION))]
    train = ds[ds["timestamp"] < test_start]
    test = ds[ds["timestamp"] >= test_start]
    X_train, y_train = train[feature_cols], train["y"]
    X_test, y_test = test[feature_cols], test["y"]

    print(f"Train: {train['timestamp'].min()} -> {train['timestamp'].max()} ({len(train):,} rows)")
    print(f"Test:  {test['timestamp'].min()} -> {test['timestamp'].max()} ({len(test):,} rows)")
    print(f"Share of 'up' bars in test set: {y_test.mean():.4f}")

    model = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds, digits=4))

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "symbols": SYMBOLS,
        "horizon_bars": HORIZON_BARS,
        "spy_symbol": SPY_SYMBOL,
        "test_start": test_start,
    }
    dump(artifact, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
