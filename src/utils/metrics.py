"""
Time-series forecasting metrics.

All functions expect:
    y: Ground truth (1D array)
    yhat: Predictions (1D array)
"""
import numpy as np

def mae(y, yhat):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(yhat - y)))

def mse(y, yhat):
    """Mean Squared Error."""
    return float(np.mean((yhat - y) ** 2))

def rmse(y, yhat):
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y, yhat)))

def smape(y, yhat, eps=1e-6):
    """
    Symmetric Mean Absolute Percentage Error (%).
    Range: [0, 200], lower is better.
    """
    denom = (np.abs(y) + np.abs(yhat) + eps) / 2.0
    return float(np.mean(np.abs(yhat - y) / denom) * 100.0)

def mape(y, yhat, eps=1e-6):
    """
    Mean Absolute Percentage Error (%).
    Asymmetric version of sMAPE.
    """
    return float(np.mean(np.abs((y - yhat) / (y + eps))) * 100.0)

def mase(y, yhat, y_train=None, seasonality=1):
    """
    Mean Absolute Scaled Error.
    
    Args:
        y: Ground truth test values (1D array)
        yhat: Predictions (1D array)
        y_train: Training data for computing naive forecast error (1D array)
                 If None, uses y itself (less accurate but workable)
        seasonality: Seasonal period (e.g., 24 for daily in hourly data)
    
    Returns:
        MASE value. <1.0 means better than seasonal naive.
    
    Reference: Hyndman & Koehler (2006)
    """
    mae_forecast = np.mean(np.abs(y - yhat))
    
    if y_train is not None:
        # Proper MASE: use training data for naive forecast
        if len(y_train) <= seasonality:
            # Fallback to non-seasonal naive
            naive_errors = np.abs(np.diff(y_train))
        else:
            # Seasonal naive: compare t with t-seasonality
            naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    else:
        # Fallback: use test data (less accurate)
        if len(y) <= seasonality:
            naive_errors = np.abs(np.diff(y))
        else:
            naive_errors = np.abs(y[seasonality:] - y[:-seasonality])
    
    mae_naive = np.mean(naive_errors)
    
    # Prevent division by zero
    if mae_naive < 1e-10:
        return float('inf') if mae_forecast > 1e-10 else 1.0
    
    return float(mae_forecast / mae_naive)

def les(err_eval, err_train):
    """
    Length Extrapolation Score.
    LES = MAE(long_context) / MAE(short_context)
    """
    return float(err_eval) / (float(err_train) + 1e-12)
