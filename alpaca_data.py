from datetime import datetime, timedelta, timezone
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest

from config import TIMEFRAME, get_alpaca_credentials


class AlpacaData:
    """Historical bar downloader used for training and backtesting."""

    def __init__(self):
        api_key, api_secret = get_alpaca_credentials()
        self.data = StockHistoricalDataClient(api_key, api_secret)

    def get_bars(self, symbol: str, start_utc: datetime, end_utc: datetime, limit: int = 10000) -> pd.DataFrame:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TIMEFRAME,
            start=start_utc,
            end=end_utc,
            limit=limit,
        )
        bars = self.data.get_stock_bars(req).df

        df = bars.reset_index()
        df = df[df["symbol"] == symbol].copy()
        df = df.sort_values("timestamp")

        # Keep only what we need (and standardize columns)
        return df[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]


def safe_end_for_free_feed() -> datetime:
    # Free data can restrict the very latest window; shift end back a bit
    return datetime.now(timezone.utc) - timedelta(minutes=16)
