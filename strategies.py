import pandas as pd


class MovingAverageCrossover:
    """
    Simple MA crossover:
      signal = +1 if fast_ma > slow_ma
      signal = -1 if fast_ma <= slow_ma
      signal = 0 if not enough data
    """

    def __init__(self, fast_window: int = 10, slow_window: int = 30):
        if fast_window >= slow_window:
            raise ValueError("fast_window must be < slow_window")

        self.fast_window = fast_window
        self.slow_window = slow_window

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["fast_ma"] = out["close"].rolling(self.fast_window).mean()
        out["slow_ma"] = out["close"].rolling(self.slow_window).mean()
        return out

    def signal(self, df: pd.DataFrame) -> int:
        df_ind = self.add_indicators(df)
        latest = df_ind.iloc[-1]

        if pd.isna(latest["fast_ma"]) or pd.isna(latest["slow_ma"]):
            return 0

        return 1 if latest["fast_ma"] > latest["slow_ma"] else -1