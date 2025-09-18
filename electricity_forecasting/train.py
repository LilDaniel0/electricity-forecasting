import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
import json

df = pd.read_pickle("data/AEP_hourly_features.pkl")

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


# === SERIALIZAR ===
reg.save_model("model/xgb_model.json")

with open("model/feature_order.json", "w") as f:
    json.dump(list(X_train.columns), f, indent=2)

config = {"target": "AEP_MW", "min_history_hours": 168, "freq": "H"}  # por usar lag_168
with open("model/config.json", "w") as f:
    json.dump(config, f, indent=2)
print("✅ Modelo y metadatos guardados en model/")
# ==================


# -------------------------------- Feature importances-------------------------------
pd.DataFrame(
    reg.feature_importances_, index=reg.feature_names_in_, columns=["importance"]
).sort_values(by="importance", ascending=False).plot(kind="bar")
# -----------------------------------------------------------------------------------
# Ver columnas sospechosas para leaked de informacion
bad = [c for c in X_train.columns if ("AEP_MW" in c) or ("prediction" in c)]
print("Cols sospechosas:", bad)  # debe imprimir lista vacía

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
