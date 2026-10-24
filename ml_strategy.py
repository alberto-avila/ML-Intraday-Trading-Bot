import pandas as pd
import numpy as np
from joblib import load

from feature_engineering import add_features


class MLProbabilityStrategy:
    """
    Long/cash strategy using a trained classifier's P(up) probability.

    Enter long if p_up >= enter_th
    Exit to cash if p_up <= exit_th
    Else hold

    Works per-symbol independently.
    """

    def __init__(self, model_path: str, enter_th: float = 0.55, exit_th: float = 0.45):
        artifact = load(model_path)
        self.model = artifact["model"]
        self.feature_cols = artifact["feature_cols"]
        self.symbols = artifact.get("symbols", None)
        self.horizon_bars = artifact.get("horizon_bars", None)

        if not (0 < exit_th < enter_th < 1):
            raise ValueError("Require 0 < exit_th < enter_th < 1")

        self.enter_th = enter_th
        self.exit_th = exit_th

    def predict_proba_up_with_context(self, symbol_df, symbol, spy_df, univ_ret_mean, univ_ret_std):
        # build symbol features
        work = symbol_df.copy()
        work["symbol"] = symbol
        work = add_features(work)

        # latest valid row
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        if work.empty:
            return None
        latest = work.iloc[[-1]].copy()

        # add universe cross-sectional features
        latest["univ_ret_mean"] = float(univ_ret_mean)
        latest["univ_ret_std"] = float(univ_ret_std)
        latest["rel_ret_1"] = latest["ret_1"] - latest["univ_ret_mean"]
        latest["ret_z"] = (latest["ret_1"] - latest["univ_ret_mean"]) / (latest["univ_ret_std"].replace(0, np.nan))

        # add SPY features (compute once per loop in main and pass latest row)
        spy_work = add_features(spy_df.copy()).replace([np.inf, -np.inf], np.nan).dropna()
        if spy_work.empty:
            return None
        spy_latest = spy_work.iloc[-1]
        latest["spy_ret_1"] = spy_latest["ret_1"]
        latest["spy_vol_8"] = spy_latest["vol_8"]
        latest["spy_vol_24"] = spy_latest["vol_24"]
        latest["spy_ma_ratio_5_15"] = (spy_latest["ma_5"] / spy_latest["ma_15"]) - 1.0
        rng = (spy_latest["high"] - spy_latest["low"]) or np.nan
        latest["spy_close_loc"] = (spy_latest["close"] - spy_latest["low"]) / rng
        latest["spy_vol_z_20"] = spy_latest["vol_z_20"] if "vol_z_20" in spy_work.columns else np.nan

        # one-hot symbol
        latest = pd.get_dummies(latest, columns=["symbol"], drop_first=False)

        # align columns
        for c in self.feature_cols:
            if c not in latest.columns:
                latest[c] = 0.0
        latest = latest[self.feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if latest.empty:
            return None

        return float(self.model.predict_proba(latest)[0, 1])

    def decide(self, p_up: float | None, current_qty: int) -> str:
        """
        Returns one of: "BUY", "SELL", "HOLD"
        """
        if p_up is None:
            return "HOLD"

        # long/cash only
        if current_qty == 0 and p_up >= self.enter_th:
            return "BUY"

        if current_qty > 0 and p_up <= self.exit_th:
            return "SELL"

        return "HOLD"