"""
Live paper-trading loop.

Every minute: pull the latest 15-min bars for the universe and SPY, and for each
symbol with a new bar, score P(next bar up) with the LightGBM model and BUY /
SELL / HOLD on the Alpaca paper account.
"""
import time
import math
from collections import defaultdict

import numpy as np
from alpaca.trading.enums import OrderSide

from alpaca_api import AlpacaClient
from config import SYMBOLS, SPY_SYMBOL, TIMEFRAME, MODEL_PATH, ALLOC_PCT, ENTER_TH, EXIT_TH
from ml_strategy import MLProbabilityStrategy


BARS_LIMIT = 400
RUN_EVERY_SECONDS = 60

last_seen_bar_ts = defaultdict(lambda: None)


def compute_shares_to_buy(equity: float, n_symbols: int, last_price: float) -> int:
    if last_price <= 0:
        return 0
    target_dollars = (equity * ALLOC_PCT) / max(n_symbols, 1)
    shares = int(math.floor(target_dollars / last_price))
    return max(shares, 0)


def run_once(client: AlpacaClient, strat: MLProbabilityStrategy):
    if not client.is_market_open():
        print("Market closed - waiting.")
        return

    acct = client.get_account()
    equity = float(acct.equity)

    print(f"Account status={acct.status}, equity={acct.equity}, buying_power={acct.buying_power}")

    # 1) Fetch SPY context once
    spy_df = client.get_recent_bars_df(SPY_SYMBOL, timeframe=TIMEFRAME, limit=BARS_LIMIT)

    # 2) Fetch all symbol data first
    symbol_dfs = {}
    for symbol in SYMBOLS:
        symbol_dfs[symbol] = client.get_recent_bars_df(symbol, timeframe=TIMEFRAME, limit=BARS_LIMIT)

    # 3) Universe cross-sectional stats (latest 1-bar return across symbols)
    latest_returns = []
    for sym, df_sym in symbol_dfs.items():
        if len(df_sym) < 2:
            continue
        ret_1 = float(df_sym["close"].pct_change().iloc[-1])
        if np.isfinite(ret_1):
            latest_returns.append(ret_1)

    # If something goes wrong (no returns), fall back to 0/very small std
    if len(latest_returns) < 2:
        univ_ret_mean = 0.0
        univ_ret_std = 1e-6
    else:
        univ_ret_mean = float(np.mean(latest_returns))
        univ_ret_std = float(np.std(latest_returns, ddof=1))  # sample std, same as pandas in training
        if univ_ret_std == 0.0:
            univ_ret_std = 1e-6

    # 4) Decide/trade per symbol
    for symbol, df in symbol_dfs.items():
        if df.empty:
            continue
        bar_ts = df["timestamp"].iloc[-1]

        # Only act once per new 15-min bar per symbol
        if last_seen_bar_ts[symbol] == bar_ts:
            continue
        last_seen_bar_ts[symbol] = bar_ts

        current_qty = client.get_position_qty(symbol)

        p_up = strat.predict_proba_up_with_context(
            symbol_df=df,
            symbol=symbol,
            spy_df=spy_df,
            univ_ret_mean=univ_ret_mean,
            univ_ret_std=univ_ret_std
        )

        decision = strat.decide(p_up, current_qty)

        p_str = "None" if p_up is None else f"{p_up:.3f}"
        print(f"{symbol} @ {bar_ts}: p_up={p_str} qty={current_qty} decision={decision}")

        if decision == "BUY":
            last_price = float(df["close"].iloc[-1])
            shares = compute_shares_to_buy(equity, len(SYMBOLS), last_price)

            if shares >= 1:
                resp = client.submit_market_order(symbol, shares, OrderSide.BUY)
                print(f"{symbol}: BUY {shares} @~{last_price:.2f} submitted {resp.id}")
            else:
                print(f"{symbol}: BUY skipped (equity too low vs price {last_price:.2f})")

        elif decision == "SELL":
            if current_qty > 0:
                resp = client.submit_market_order(symbol, current_qty, OrderSide.SELL)
                print(f"{symbol}: SELL {current_qty} submitted {resp.id}")


def main():
    client = AlpacaClient(paper=True)
    strat = MLProbabilityStrategy(MODEL_PATH, enter_th=ENTER_TH, exit_th=EXIT_TH)

    while True:
        try:
            run_once(client, strat)
        except Exception as e:
            print("Error:", e)

        time.sleep(RUN_EVERY_SECONDS)


if __name__ == "__main__":
    main()
