import pandas as pd


def synchronize_modalities(patient_data: dict, freq: str = '5min') -> pd.DataFrame:
    """
    Synchronize CGM, insulin, carbohydrate data onto a uniform 5-minute time grid.
    """
    df_g = patient_data['glucose']
    df_i = patient_data.get('insulin', pd.DataFrame())
    df_c = patient_data.get('carbs', pd.DataFrame())

    if df_g.empty:
        return pd.DataFrame()

    df_g = df_g.set_index('timestamp')
    # Rename to standard col name
    glucose_col = [c for c in df_g.columns if c != 'timestamp'][0]
    df_g = df_g.rename(columns={glucose_col: 'glucose_level'})

    start_time = df_g.index.min()
    end_time = df_g.index.max()
    time_grid = pd.date_range(start=start_time, end=end_time, freq=freq)

    sync_g = df_g['glucose_level'].resample(freq).mean()

    result = pd.DataFrame(index=time_grid)
    result.index.name = 'timestamp'
    result = result.join(sync_g)

    if not df_i.empty and 'timestamp' in df_i.columns:
        df_i = df_i.set_index('timestamp')
        insulin_col = [c for c in df_i.columns][0]
        df_i = df_i.rename(columns={insulin_col: 'bolus'})
        sync_i = df_i['bolus'].resample(freq).sum()
        result = result.join(sync_i)
    else:
        result['bolus'] = 0.0

    if not df_c.empty and 'timestamp' in df_c.columns:
        df_c = df_c.set_index('timestamp')
        carb_col = [c for c in df_c.columns][0]
        df_c = df_c.rename(columns={carb_col: 'meal'})
        sync_c = df_c['meal'].resample(freq).sum()
        result = result.join(sync_c)
    else:
        result['meal'] = 0.0

    result['bolus'] = result['bolus'].fillna(0.0)
    result['meal'] = result['meal'].fillna(0.0)
    return result
