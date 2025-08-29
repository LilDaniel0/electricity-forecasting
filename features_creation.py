import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

palette = sns.color_palette()
plt.style.use("fivethirtyeight")

# configuracion inicial---------------
df = pd.read_csv("data/AEP_hourly.csv")
df = df.set_index("Datetime")
df.index = pd.to_datetime(df.index)
df = df.sort_index()
# ------------------------------------

df.info()
df.describe()
df.plot(style=".", figsize=(15, 5), color=palette, title="AEP")

# Suma diaria / semanal (semana empezando lunes) / mensual (inicio de mes)
aep_d = df["AEP_MW"].resample("D").sum()
aep_w = df["AEP_MW"].resample("W-MON").sum()
aep_m = df["AEP_MW"].resample("MS").sum()

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

df.to_pickle("data/AEP_hourly_features.pkl")

#####################################################################
############# ----------- VISUALIZATION -------------#############
#####################################################################


df_weekly = df.resample("W").mean()
df_monthly = df.resample("M").mean()

df_weekly[["AEP_MW"]].plot(figsize=(15, 5), title="Weekly mean")
df_monthly[["AEP_MW"]].plot(figsize=(15, 5), title="Monthly mean")


# --Visualizaicon de los nuevos features---------------------------
df[["AEP_MW", "roll4_mean", "roll24_mean"]].query(
    "index < '2017-05-8' & index >= '2017-04-01'"
).plot(figsize=(15, 5), title="Week of Data")


df[["roll168_mean"]].query("index < '2017-05-08' & index >= '2016-04-01'").plot(
    figsize=(15, 5), title="Week of Data"
)


# Visualizar las tendencias por escala de tiempo
fig, ax = plt.subplots(
    nrows=3,
    sharex=False,
    figsize=(20, 13),
)
aep_d.plot(style=".", color=palette, title="AEP Diario", ax=ax[0])
aep_w.plot(color=palette, title="AEP Semanal", ax=ax[1])
aep_m.plot(color=palette, title="AEP Mensual", ax=ax[2])
ax[0].set_xlabel("")
ax[1].set_xlabel("")
plt.show()


# Visualizar una sola semana de data - dos formas de hacer lo mismo
df[["AEP_MW", "prediction"]].query("index < '2017-03-08' & index >= '2017-03-01'").plot(
    figsize=(15, 5), title="Week of Data"
)
df.loc[(df.index >= "2017-01-01") & (df.index < "2017-01-08")].plot(
    figsize=(15, 5), title="Week of Data"
)

# Visualizar mayor uso de energia por peridos de tiempo

periodo = "month"
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(df, x=periodo, y="AEP_MW", palette="Blues", ax=ax)  # Per {periodo}
plt.title(f"Mayor uso de energia por {periodo}")
plt.show()
