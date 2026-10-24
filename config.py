"""
Shared settings for training, backtesting and live trading.
Keeping them in one place guarantees the three stages use the same universe,
bar size and decision thresholds.
"""
import os

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN",     # tech
    "LMT", "CAT",                       # industrial/defense
    "BAC", "JPM",                       # financials
    "XOM",                              # energy
    "JNJ",                              # healthcare
    "PG", "KO",                         # consumer staples
    "BTC", "GLD", "USO",                # bitcoin, gold and oil ETFs
]
SPY_SYMBOL = "SPY"  # market-regime context

TIMEFRAME = TimeFrame(15, TimeFrameUnit.Minute)
MODEL_PATH = "lgbm_15m.joblib"

ALLOC_PCT = 0.90  # invest up to 90% of equity, split equally across symbols
ENTER_TH = 0.55   # go long when P(up) >= ENTER_TH
EXIT_TH = 0.45    # go to cash when P(up) <= EXIT_TH


def get_alpaca_credentials() -> tuple[str, str]:
    """Read API keys from environment variables (or a local .env file)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY / ALPACA_API_SECRET. "
            "Copy .env.example to .env and fill in your paper-trading keys."
        )
    return api_key, api_secret
