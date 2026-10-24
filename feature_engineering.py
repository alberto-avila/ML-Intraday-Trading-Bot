import numpy as np
import pandas as pd

SPY_FEATURES = ["ret_1", "vol_8", "vol_24", "ma_ratio_5_15", "close_loc", "vol_z_20"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: timestamp, open, high, low, close, volume
    Returns df with features, no leakage (uses past data only).
    """
    out = df.copy()

    # Returns
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_2"] = out["close"].pct_change(2)
    out["ret_4"] = out["close"].pct_change(4)
    out["ret_8"] = out["close"].pct_change(8)

    # Rolling volatility
    out["vol_8"]  = out["ret_1"].rolling(8).std()
    out["vol_24"] = out["ret_1"].rolling(24).std()

    # Moving averages (trend)
    out["ma_5"]  = out["close"].rolling(5).mean()
    out["ma_15"] = out["close"].rolling(15).mean()
    out["ma_ratio_5_15"] = (out["ma_5"] / out["ma_15"]) - 1.0

    # Price position in range (simple regime-ish feature)
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_loc"] = (out["close"] - out["low"]) / rng

    # Volume features
    out["vol_mean_20"] = out["volume"].rolling(20).mean()
    out["vol_z_20"] = (out["volume"] - out["vol_mean_20"]) / out["volume"].rolling(20).std()

    # Clean up inf/nan created by rolling/std
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def add_spy_features(panel: pd.DataFrame, spy_raw: pd.DataFrame) -> pd.DataFrame:
    """Merge SPY (market regime) features onto every symbol row by timestamp."""
    spy_feat = add_features(spy_raw)[["timestamp"] + SPY_FEATURES].copy()
    spy_feat = spy_feat.rename(columns={c: f"spy_{c}" for c in SPY_FEATURES})
    spy_feat = spy_feat.dropna()
    return panel.merge(spy_feat, on="timestamp", how="inner")


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compare each symbol's latest return with the rest of the universe at the same timestamp."""
    cross = (
        panel.groupby("timestamp")["ret_1"]
             .agg(univ_ret_mean="mean", univ_ret_std="std")
             .reset_index()
    )
    out = panel.merge(cross, on="timestamp", how="left")

    # Relative / zscore return vs peers
    out["rel_ret_1"] = out["ret_1"] - out["univ_ret_mean"]
    out["ret_z"] = (out["ret_1"] - out["univ_ret_mean"]) / (out["univ_ret_std"].replace(0, np.nan))
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def align_to_model(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """One-hot encode the symbol and order columns exactly as the model saw them in training."""
    X = pd.get_dummies(panel, columns=["symbol"], drop_first=False)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    return X[feature_cols]
