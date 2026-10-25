import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, MARD, and R² for continuous glucose predictions.

    RMSE  = sqrt(mean((y_true - y_pred)^2))
    MAE   = mean(|y_true - y_pred|)
    MARD  = mean(|y_true - y_pred| / |y_true|) * 100    [zero-safe]
    R²    = coefficient of determination
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Mask to avoid division by zero in MARD
    mask = y_true > 1e-6
    y_t = y_true[mask]
    y_p = y_pred[mask]

    if len(y_t) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'MARD': np.nan, 'R2': np.nan}

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    mard = float(np.mean(np.abs(y_t - y_p) / np.abs(y_t)) * 100)
    r2 = float(r2_score(y_t, y_p))

    return {'RMSE': rmse, 'MAE': mae, 'MARD': mard, 'R2': r2}
