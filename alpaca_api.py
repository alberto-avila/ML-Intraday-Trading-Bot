import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, PositionSide

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest

from datetime import datetime, timedelta, timezone

from config import get_alpaca_credentials


class AlpacaClient:
    """
    Wraps Alpaca Trading + Market Data clients.
    Paper trading by default.
    """

    def __init__(self, paper: bool = True):
        api_key, api_secret = get_alpaca_credentials()

        self.trading = TradingClient(api_key, api_secret, paper=paper)
        self.data = StockHistoricalDataClient(api_key, api_secret)

    # ---------- Account / Positions ----------
    def get_account(self):
        return self.trading.get_account()

    def is_market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    def get_position_qty(self, symbol: str) -> int:
        try:
            pos = self.trading.get_open_position(symbol)
            return int(float(pos.qty))  # qty comes as string
        except Exception:
            return 0

    # ---------- Market Data ----------
    def get_recent_bars_df(
        self,
        symbol: str,
        timeframe,
        limit: int = 300,
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        # Free data can restrict the most recent ~15 minutes; ending a bit earlier avoids errors.
        end = datetime.now(timezone.utc) - timedelta(minutes=16)
        # Without an explicit start, Alpaca only returns bars from the current day,
        # which is not enough history for the rolling features.
        start = end - timedelta(days=lookback_days)

        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start,
            end=end,
        )
        bars = self.data.get_stock_bars(req).df
        df = bars.reset_index()
        df = df[df["symbol"] == symbol].copy()
        df = df.sort_values("timestamp")
        return df.tail(limit)

    # ---------- Orders ----------
    def submit_market_order(self, symbol: str, qty: int, side: OrderSide):
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        return self.trading.submit_order(order)

    def liquidate_all_positions_market(self) -> list:
        """
        Closes every open position with a market order.
        - LONG => SELL qty
        - SHORT => BUY qty (cover)
        Returns list of submitted order responses.
        """
        positions = self.trading.get_all_positions()
        orders = []

        for pos in positions:
            symbol = pos.symbol
            qty = int(float(pos.qty))  # qty is string
            if qty == 0:
                continue

            # pos.side is a PositionSide enum in alpaca-py
            if pos.side == PositionSide.LONG:
                side = OrderSide.SELL
            elif pos.side == PositionSide.SHORT:
                side = OrderSide.BUY
            else:
                print(f"WARNING: Unknown position side for {symbol}: {pos.side}. Skipping.")
                continue

            resp = self.submit_market_order(symbol, abs(qty), side)
            orders.append(resp)
            print(f"Liquidation order: {symbol} {side.value} {abs(qty)} (order_id={resp.id})")

        return orders
