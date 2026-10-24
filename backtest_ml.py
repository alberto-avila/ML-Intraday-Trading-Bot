"""
Backtest the trained model on out-of-sample data.

By default the backtest starts where the training script's held-out test set
starts (saved in the model artifact), so the model never trades on bars it was
fit on. It applies the same long/cash rules as the live bot: only acts during
regular market hours, buys an equal-weight slice of equity when P(up) >= ENTER_TH
and exits to cash when P(up) <= EXIT_TH.
"""
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
from joblib import load

from alpaca_data import AlpacaData, safe_end_for_free_feed
from config import SYMBOLS, SPY_SYMBOL, MODEL_PATH, ALLOC_PCT, ENTER_TH, EXIT_TH
from feature_engineering import add_features, add_spy_features, add_cross_sectional_features, align_to_model


INITIAL_CAPITAL = 100000
WARMUP_DAYS = 5  # extra history so rolling features are ready on the first backtest bar


def is_regular_hours(ts: pd.Timestamp) -> bool:
    t = ts.tz_convert("America/New_York")
    return t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0)


def compute_shares(equity, cash, n_symbols, price):
    target = (equity * ALLOC_PCT) / n_symbols
    return int(np.floor(min(target, cash) / price))


def summarize(equity: pd.Series) -> dict:
    """ROI, annualized Sharpe (from daily returns) and max drawdown of an equity curve."""
    daily = equity.groupby(equity.index.tz_convert("America/New_York").date).last()
    daily_ret = daily.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else np.nan
    drawdown = (1 - equity / equity.cummax()).max()
    return {
        "final": equity.iloc[-1],
        "roi": equity.iloc[-1] / equity.iloc[0] - 1,
        "sharpe": sharpe,
        "max_dd": drawdown,
    }


def build_backtest_panel(data: AlpacaData, start, end) -> pd.DataFrame:
    """Same feature pipeline as train_lgbm.py, minus the labels."""
    symbol_dfs = []
    for sym in SYMBOLS:
        raw = data.get_bars(sym, start_utc=start, end_utc=end)
        symbol_dfs.append(add_features(raw).dropna())
    panel = pd.concat(symbol_dfs, ignore_index=True)

    spy_raw = data.get_bars(SPY_SYMBOL, start_utc=start, end_utc=end)
    panel = add_spy_features(panel, spy_raw)
    panel = add_cross_sectional_features(panel)
    return panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def run_backtest(panel: pd.DataFrame, enter_th: float, exit_th: float):
    cash = INITIAL_CAPITAL
    positions = {s: 0 for s in SYMBOLS}
    last_price = {}
    trades = 0
    equity_curve = {}

    for t, slice_df in panel.groupby("timestamp", sort=True):
        for row in slice_df.itertuples(index=False):
            last_price[row.symbol] = row.close

        if is_regular_hours(t):
            equity = cash + sum(positions[s] * last_price.get(s, 0.0) for s in SYMBOLS)
            for row in slice_df.itertuples(index=False):
                qty = positions[row.symbol]

                if qty == 0 and row.p_up >= enter_th:
                    shares = compute_shares(equity, cash, len(SYMBOLS), row.close)
                    if shares > 0:
                        cash -= shares * row.close
                        positions[row.symbol] = shares
                        trades += 1

                elif qty > 0 and row.p_up <= exit_th:
                    cash += qty * row.close
                    positions[row.symbol] = 0
                    trades += 1

        # Mark to market with the latest known price of every symbol
        equity_curve[t] = cash + sum(positions[s] * last_price.get(s, 0.0) for s in SYMBOLS)

    return pd.Series(equity_curve), trades


def buy_and_hold(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight buy & hold of the same universe over the same period (benchmark)."""
    prices = panel.pivot(index="timestamp", columns="symbol", values="close").ffill().bfill()
    return INITIAL_CAPITAL * (prices / prices.iloc[0]).mean(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enter", type=float, default=ENTER_TH, help="P(up) threshold to enter a long")
    parser.add_argument("--exit", type=float, default=EXIT_TH, help="P(up) threshold to exit to cash")
    parser.add_argument("--days", type=int, default=None,
                        help="Backtest the last N days instead of the held-out test period")
    args = parser.parse_args()

    artifact = load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    end = safe_end_for_free_feed()
    if args.days is not None or "test_start" not in artifact:
        bt_start = pd.Timestamp(end - timedelta(days=args.days or 30))
    else:
        bt_start = artifact["test_start"]

    panel = build_backtest_panel(AlpacaData(), bt_start - timedelta(days=WARMUP_DAYS), end)
    panel = panel[panel["timestamp"] >= bt_start].copy()
    panel["p_up"] = model.predict_proba(align_to_model(panel, feature_cols))[:, 1]

    equity, trades = run_backtest(panel, args.enter, args.exit)
    strat = summarize(equity)
    bench = summarize(buy_and_hold(panel))

    print(f"\nBacktest {panel['timestamp'].min()} -> {panel['timestamp'].max()}")
    print(f"Thresholds: enter >= {args.enter}, exit <= {args.exit}")
    print("-" * 52)
    print(f"{'':16}{'ML strategy':>18}{'Buy & hold (EW)':>18}")
    print(f"{'Final equity':16}{strat['final']:>18,.2f}{bench['final']:>18,.2f}")
    print(f"{'ROI':16}{strat['roi']:>17.2%} {bench['roi']:>17.2%}")
    print(f"{'Sharpe (ann.)':16}{strat['sharpe']:>18.2f}{bench['sharpe']:>18.2f}")
    print(f"{'Max drawdown':16}{strat['max_dd']:>17.2%} {bench['max_dd']:>17.2%}")
    print(f"{'Trades':16}{trades:>18}")


if __name__ == "__main__":
    main()
