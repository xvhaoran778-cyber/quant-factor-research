import pandas as pd


def filter_universe(frame: pd.DataFrame, min_listed_days: int = 120) -> pd.DataFrame:
    required = ["is_st", "suspended", "listed_days", "pe", "roe", "growth", "close"]
    valid = frame.dropna(subset=required)
    return valid[(~valid["is_st"]) & (~valid["suspended"]) & (valid["listed_days"] >= min_listed_days) & (valid["pe"] > 0)]


def rank_multifactor(frame: pd.DataFrame, lookback: pd.DataFrame) -> pd.DataFrame:
    current = filter_universe(frame).copy()
    if current.empty:
        return current.assign(score=pd.Series(dtype=float))
    momentum = lookback.sort_values("date").groupby("symbol")["close"].agg(lambda values: values.iloc[-1] / values.iloc[0] - 1 if len(values) > 1 else 0)
    volatility = lookback.groupby("symbol")["close"].apply(lambda values: values.pct_change().std()).fillna(0)
    current["momentum"] = current["symbol"].map(momentum).fillna(0)
    current["volatility"] = current["symbol"].map(volatility).fillna(0)
    factors = {"pe": -0.20, "roe": 0.25, "growth": 0.20, "momentum": 0.25, "volatility": -0.10}
    current["score"] = 0.0
    for column, weight in factors.items():
        percentile = current[column].rank(pct=True, method="average")
        current["score"] += percentile * weight
    return current.sort_values(["score", "symbol"], ascending=[False, True])

