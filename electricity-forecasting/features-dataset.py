# features.py
import pandas as pd

# configuracion inicial---------------
df = pd.read_csv("../data/AEP_hourly.csv", parse_dates=["Datetime"]).set_index(
    "Datetime"
)
#####################################################################
############# ----------- FEATURE CREATION -------------#############
#####################################################################


def Timefeature_creation(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    # ------ creacion de lags - memoria del pasado, mismos datos desfasados
    df["AEP_MW_lag_1"] = df["AEP_MW"].shift(1)
    df["AEP_MW_lag_24"] = df["AEP_MW"].shift(24)
    df["AEP_MW_lag_168"] = df["AEP_MW"].shift(168)
    # ------ roll means std
    df["volatility_24h"] = df["AEP_MW"].shift(1).rolling(24).std()
    df["volatility_168h"] = df["AEP_MW"].shift(1).rolling(168).std()
    df["roll8_mean"] = df["AEP_MW"].shift(1).rolling(8).mean()
    df["roll24_mean"] = df["AEP_MW"].shift(1).rolling(24).mean()
    df["roll168_mean"] = df["AEP_MW"].shift(1).rolling(168).mean()
    df["roll24_min"] = df["AEP_MW"].shift(1).rolling(24).min()
    df["roll24_max"] = df["AEP_MW"].shift(1).rolling(24).max()
    df["roll168_min"] = df["AEP_MW"].shift(1).rolling(168).min()
    df["roll168_max"] = df["AEP_MW"].shift(1).rolling(168).max()

    return df


df = Timefeature_creation(df)
df.to_pickle("../data/AEP_hourly_features.pkl")
