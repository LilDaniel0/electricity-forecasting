#####################################################################
################## ----------- FUNCTIONS -------------###############
#####################################################################
import pandas as pd
import matplotlib.pyplot as plt
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split


def Preprocess(df, resample: str):

    df = df.set_index("Datetime")
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={df.columns[0]: "target"}).sort_index()
    df = df["target"].resample("M").mean()
    df.index = df.index.to_period("M")
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


def Regressor_split(df, date_test, date_val):

    split_test = date_test  # Ejemplo: "2017-01-01"
    split_val = date_val  # Ejemplo: "2016-09-01"

    train = df[df.index < split_val]
    val = df[(df.index < split_test) & (df.index >= split_val)]
    test = df[df.index >= split_test]

    # Visualizar como se hizo el split
    fig, ax = plt.subplots(figsize=(10, 5))
    train.plot(ax=ax)
    val.plot(ax=ax)
    test.plot(ax=ax)
    plt.legend(["train", "validation", "test"])
    plt.show()

    X_train, y_train = train.drop(columns="target"), train["target"]
    X_val, y_val = val.drop(columns="target"), val["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def Theta_prediction(df, train_size: int, sp: int):

    y_train, y_test = temporal_train_test_split(df, train_size=train_size)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = ThetaForecaster(sp=sp)
    forecaster.fit(y_train)

    prediction = forecaster.predict(fh)

    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(ax=ax)
    prediction.plot(ax=ax)
    plt.legend(["real", "prediction"])
    plt.title("Full DataFrame vs Theta Prediction")

    return fig
