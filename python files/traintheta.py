# train_theta.py
import pandas as pd
import joblib
from pathlib import Path
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting import plot_series

# Configuracion inicial ---------------------
MODEL = Path("../model/theta forecaster")

df = pd.read_csv("../data/AEP_hourly.csv", parse_dates=["Datetime"]).set_index(
    "Datetime"
)
y = df["AEP_MW"].resample("M").mean()
y.index = y.index.to_period("M")


#################################################################
################# Entrenamiento del modelo ######################
#################################################################
forecaster = ThetaForecaster(sp=12)
forecaster.fit(y)  # ENTRENAMOS CON TODO EL HISTÓRICO

joblib.dump(forecaster, MODEL / "theta_forecaster.joblib")
(MODEL / "theta_meta.txt").write_text(
    "resample=M (mean)\nindex=Period[M]\nmodel=Theta(sp=12)\n"
)

print("✅ Guardado model/theta_forecaster.joblib")
