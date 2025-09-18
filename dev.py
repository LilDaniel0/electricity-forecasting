import pandas as pd
import numpy as np
import electricity_forecasting.functions as fc
import matplotlib.pyplot as plt
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split


df = pd.read_csv("data/raw/COMED_hourly.csv")

#####################################################################
################# ----------- PREPROCESS  -------------##############
#####################################################################


def Preprocess(df, resample: str):

    df = df.set_index("Datetime")
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={df.columns[0]: "target"}).sort_index()
    df.index = df.index.resample(resample).mean()
    df.index = df.index.to_period(resample)
    return df


df = Preprocess(df, "M")


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
    plt.show()

    return prediction


df.duplicated()

Theta_prediction(df, 0.8, 12)
