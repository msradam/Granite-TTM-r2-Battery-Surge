"""NYC storm-surge nowcast at NOAA station 8518750.

>>> from granite_ttm_battery_surge import load_finetune, fetch_residual_series
>>> ft = load_finetune({"context_steps": 1024, "horizon_steps": 96})
>>> _, history = fetch_residual_series("8518750", "20250101", "20250215", hourly=True)
>>> forecast = ft.predict(history[-1024:].astype("float32"), horizon=96)
"""

from .data import (
    DEFAULT_STATION,
    fetch_residual_series,
    fetch_window,
    load_finetune,
)

__version__ = "0.1.0"
__all__ = ["DEFAULT_STATION", "fetch_residual_series", "fetch_window", "load_finetune"]
