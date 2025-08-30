import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# configuracion inicial---------------
df = pd.read_pickle("data/AEP_hourly_features.pkl")
palette = sns.color_palette()
plt.style.use("fivethirtyeight")

#####################################################################
############# ----------- VISUALIZATION -------------#############
#####################################################################
# ------------------------------------
df.plot(style=".", figsize=(15, 5), color=palette, title="AEP")

# Suma diaria / semanal (semana empezando lunes) / mensual (inicio de mes)
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
df_weekly.plot(style=".", color=palette, title="AEP Semanal", ax=ax[0])
df_monthly.plot(color=palette, title="AEP Mensual", ax=ax[1])
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
