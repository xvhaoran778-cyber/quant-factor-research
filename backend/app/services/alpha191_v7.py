"""Minimal alpha191_v7 — just load_v7_panel for V8 dependency."""
from pathlib import Path
import pandas as pd

def load_v7_panel(store, base_panel):
    cache_dir = Path(store.root) / "alpha191_v4_annual" / "factor_cache"
    path = cache_dir / "alpha_132.parquet"
    values = pd.read_parquet(path, columns=["date", "symbol", "factor"])
    values["date"] = pd.to_datetime(values["date"])
    values = values.rename(columns={"factor": "alpha_132"})
    merged = base_panel.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    return merged.merge(values, on=["date", "symbol"], how="left")
