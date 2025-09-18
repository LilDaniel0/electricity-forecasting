import pandas as pd
import electricity_forecasting.functions as fc

df = pd.read_csv("data/raw/COMED_hourly.csv")

#####################################################################
################# ----------- PREPROCESS  -------------##############
#####################################################################


df = fc.Preprocess(df, "M")

fc.Theta_prediction(df, 0.8, 12)
