import pandas as pd
import numpy as np


df = pd.read_csv("data/raw/COMED_hourly.csv")


def Preprocess(df):

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.set_index("Datetime").resample("H").mean()
    df = df.rename(columns={df.columns[0]: "target"})
    return df


Preprocess(df)
