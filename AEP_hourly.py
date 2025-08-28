import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.utils.plotting import plot_series
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

palette = sns.color_palette()
plt.style.use("fivethirtyeight")

# configuracion inicial---------------
df = pd.read_csv("data/AEP_hourly.csv")
df = df.set_index("Datetime")
df.index = pd.to_datetime(df.index)
df = df.sort_index()
# ------------------------------------

df.info()
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

#####################################################################
############# ----------- TEST / SPLIT DATA -------------############
#####################################################################

split_test = "2017-01-01"
split_val = "2016-09-01"

train = df[df.index < split_val]
val = df[(df.index < split_test) & (df.index >= split_val)]
test = df[df.index >= split_test]

X_train, y_train = train.drop(columns="AEP_MW"), train["AEP_MW"]
X_val, y_val = val.drop(columns="AEP_MW"), val["AEP_MW"]
X_test, y_test = test.drop(columns="AEP_MW"), test["AEP_MW"]


######################################################################
############# ----------- MODELING / PREDICTION ---------#############
######################################################################

reg = xgb.XGBRegressor(n_estimators=5000, early_stopping_rounds=200, learning_rate=0.05)

reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

X_test["prediction"] = reg.predict(X_test)  # PREDICTION

mean_absolute_percentage_error(y_test, X_test["prediction"])

# -------------------------------- Feature importances-------------------------------
pd.DataFrame(
    reg.feature_importances_, index=reg.feature_names_in_, columns=["importance"]
).sort_values(by="importance", ascending=False).plot(kind="bar")
# -----------------------------------------------------------------------------------
# Ver columnas sospechosas para leaked de informacion
bad = [c for c in X_train.columns if ("AEP_MW" in c) or ("prediction" in c)]
print("Cols sospechosas:", bad)  # debe imprimir lista vacía


######################### Theta forecasting #########################

y = df_monthly[["AEP_MW"]]

y_temp_train, y_temp_test = temporal_train_test_split(y, train_size=0.8)
fh = ForecastingHorizon(y_temp_test.index, is_relative=False)
forecaster = ThetaForecaster(sp=12)
forecaster.fit(y_temp_train)


######################################################################
########### -------- VISUALIZATION OF PREDICTION ---------############
######################################################################
#
df = df.merge(X_test[["prediction"]], how="left", left_index=True, right_index=True)

fig, ax = plt.subplots(figsize=(40, 10))
df[["AEP_MW", "prediction"]].plot(ax=ax, style=".")
ax.legend(["Real", "Prediction"])
plt.show()

# --------------------- Visualizaicon de prediccion por fechas  ---------------------------
df[["AEP_MW", "prediction"]].query("index < '2017-04-8' & index >= '2017-04-01'").plot(
    figsize=(15, 5), title="Week of Data"
)
plt.show()
