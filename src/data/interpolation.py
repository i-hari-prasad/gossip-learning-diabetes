import pandas as pd
import numpy as np


def linear_interpolation(df: pd.DataFrame, glucose_col: str = 'glucose_level',
                         max_gap: int = 6) -> tuple:
    """
    Linear time-based interpolation for missing glucose values.
    G_t = G_t1 + ((t-t1)/(t2-t1)) * (G_t2 - G_t1)
    """
    df_interp = df.copy()
    missing_before = int(df_interp[glucose_col].isna().sum())
    df_interp[glucose_col] = df_interp[glucose_col].interpolate(method='time', limit=max_gap)
    missing_after = int(df_interp[glucose_col].isna().sum())
    recovered = missing_before - missing_after
    stats = {
        'missing_before': missing_before,
        'missing_after': missing_after,
        'recovered': recovered,
        'recovery_percent': float(recovered / max(1, missing_before) * 100)
    }
    return df_interp, stats
