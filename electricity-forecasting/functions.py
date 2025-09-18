#####################################################################
############# ----------- FEATURE CREATION -------------#############
#####################################################################
import pandas as pd


def Preprocess(df):

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.set_index("Datetime").resample("H").mean()
    df = df.rename(columns={df.columns[0]: "target"})
    return df


def Timefeature_creation(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    # ------ creacion de lags - memoria del pasado, mismos datos desfasados
    df["target_lag_1"] = df["target"].shift(1)
    df["target_lag_24"] = df["target"].shift(24)
    df["target_lag_168"] = df["target"].shift(168)
    # ------ roll means std
    df["volatility_24h"] = df["target"].shift(1).rolling(24).std()
    df["volatility_168h"] = df["target"].shift(1).rolling(168).std()
    df["roll8_mean"] = df["target"].shift(1).rolling(8).mean()
    df["roll24_mean"] = df["target"].shift(1).rolling(24).mean()
    df["roll168_mean"] = df["target"].shift(1).rolling(168).mean()
    df["roll24_min"] = df["target"].shift(1).rolling(24).min()
    df["roll24_max"] = df["target"].shift(1).rolling(24).max()
    df["roll168_min"] = df["target"].shift(1).rolling(168).min()
    df["roll168_max"] = df["target"].shift(1).rolling(168).max()

    return df
