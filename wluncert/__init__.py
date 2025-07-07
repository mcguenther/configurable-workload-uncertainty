"""Top-level package for wluncert.

This module lazily imports heavy dependencies. If optional packages like
TensorFlow are missing, :class:`DaLRegressor` will simply not be available.
"""

try:
    from .dal import DaLRegressor
except Exception:  # pragma: no cover - optional dependency not installed
    DaLRegressor = None
