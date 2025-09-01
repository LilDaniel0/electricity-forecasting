# features.py
import pandas as pd
import numpy as np


def make_features_for_next_step(history: pd.Series, when: pd.Timestamp):
    """
    history: Serie con índice datetime y valores del target (AEP_MW),
             DEBE llegar hasta when - 1h.
    when:    timestamp (hora a predecir)

    Devuelve: DataFrame de UNA fila con las features
    """
    # checks simples
    if history.index.max() != when - pd.Timedelta(hours=1):
        raise ValueError("El history debe terminar exactamente en when-1h")

    # features de calendario
    hour = when.hour
    dayofweek = when.dayofweek
    month = when.month

    # lags (asumen que history tiene ≥168 horas)
    lag_1 = history.iloc[-1]
    lag_24 = history.iloc[-24] if len(history) >= 24 else np.nan
    lag_168 = history.iloc[-168] if len(history) >= 168 else np.nan

    # rollings
    roll24_mean = history.iloc[-24:].mean() if len(history) >= 24 else np.nan
    roll168_mean = history.iloc[-168:].mean() if len(history) >= 168 else np.nan
    roll24_min = history.iloc[-24:].min() if len(history) >= 24 else np.nan
    roll24_max = history.iloc[-24:].max() if len(history) >= 24 else np.nan

    row = {
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "AEP_MW_lag_1": lag_1,
        "AEP_MW_lag_24": lag_24,
        "AEP_MW_lag_168": lag_168,
        "roll24_mean": roll24_mean,
        "roll168_mean": roll168_mean,
        "roll24_min": roll24_min,
        "roll24_max": roll24_max,
    }
    return pd.DataFrame([row], index=[when])
