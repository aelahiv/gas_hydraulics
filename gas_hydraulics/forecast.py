"""Simple forecasting utilities used by the plugin.

Pure Python so it can be tested without QGIS.
"""
from __future__ import annotations
from typing import List
import statistics


def moving_average_forecast(data: List[float], steps: int = 1, window: int = 3) -> List[float]:
    """Compute a naive moving average forecast.

    data: historical numeric series (list of floats)
    steps: number of future points to forecast
    window: size of moving window to compute average
    """
    if not data:
        raise ValueError("data must be a non-empty list")
    if window <= 0:
        raise ValueError("window must be positive")
    forecasts: List[float] = []
    series = list(data)
    for _ in range(steps):
        window_vals = series[-window:] if len(series) >= window else series
        avg = statistics.mean(window_vals)
        forecasts.append(avg)
        series.append(avg)
    return forecasts


def linear_regression_forecast(data: List[float], steps: int = 1) -> List[float]:
    """Compute a simple linear regression (OLS) forecast using index as x.

    Returns list of forecasted values for the next `steps` indices.
    """
    if not data:
        raise ValueError("data must be a non-empty list")
    n = len(data)
    xs = list(range(n))
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(data)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        # All x the same (single data point). Return repeating value.
        return [data[-1]] * steps
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, data))
    slope = num / denom
    intercept = y_mean - slope * x_mean
    forecasts = []
    for i in range(n, n + steps):
        forecasts.append(intercept + slope * i)
    return forecasts
