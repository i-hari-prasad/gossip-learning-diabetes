import numpy as np
import pandas as pd


def create_sliding_windows(df: pd.DataFrame, window_size: int,
                           prediction_horizon: int, stride: int = 1):
    """
    Convert synchronized time series into supervised learning sequences.

    Input window: [t-W+1, ..., t]
    Target:       glucose[t + horizon]

    Args:
        df: Synchronized DataFrame with DatetimeIndex.
        window_size: Number of past time steps in each input window.
        prediction_horizon: Steps ahead to predict.
        stride: Step size for the sliding window.

    Returns:
        X: np.ndarray of shape (samples, window_size, features)
        Y: np.ndarray of shape (samples,) — future glucose (scaled)
    """
    # Only use rows where glucose_level is available
    valid_df = df.dropna(subset=['glucose_level'])

    if len(valid_df) < window_size + prediction_horizon:
        return np.array([]), np.array([])

    data = valid_df.values.astype(np.float32)
    glucose_idx = list(valid_df.columns).index('glucose_level')

    # Infer expected time step from index
    if len(valid_df) > 1:
        diffs = valid_df.index.to_series().diff().dropna()
        expected_step = diffs.mode()[0]
    else:
        return np.array([]), np.array([])

    X, Y = [], []
    for i in range(0, len(valid_df) - window_size - prediction_horizon + 1, stride):
        window_end = i + window_size
        target_pos = window_end + prediction_horizon - 1

        # Enforce strict time continuity — no gaps inside the window
        window_times = valid_df.index[i:window_end]
        target_time = valid_df.index[target_pos]

        # Check window continuity
        window_diffs = pd.Series(window_times).diff().dropna()
        if not (window_diffs == expected_step).all():
            continue

        # Check gap between window end and target
        gap = target_time - valid_df.index[window_end - 1]
        if gap != expected_step * prediction_horizon:
            continue

        X.append(data[i:window_end, :])
        Y.append(data[target_pos, glucose_idx])

    if not X:
        return np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
