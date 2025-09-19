import streamlit as st
from pathlib import Path
import pandas as pd
import electricity_forecasting.functions as fc
import time

data = Path("data/to_predict")


# Session state
st.session_state.file_uploaded = False


st.title("Energy Predictor")
st.markdown("Make your predictions here!")

st.divider()

with st.sidebar:
    if st.button("clear"):
        items = list(data.iterdir())
        for item in items:
            try:
                item.unlink()
                st.session_state.file_uploaded = False
            except Exception as e:
                st.error(f"There's an error: {e}")


file_uploader = st.file_uploader(
    "Upload the CSV file that contais the energy information", type=["csv"]
)
if file_uploader is not None:
    with open(data / "data.csv", "wb") as f:
        f.write(file_uploader.getvalue())
        st.session_state.file_uploaded = True

if st.session_state.file_uploaded is True:
    mode = st.radio(
        "Select a mode:",
        options=[
            "ThetaForecasting",
            "XGBRegressor",
        ],
    )


if st.session_state.file_uploaded is True:
    if st.button("Press me to get the Prediction"):
        if mode is "ThetaForecasting":
            with st.spinner("Wait for it..."):
                df = pd.read_csv(data / "data.csv")
                df = fc.Preprocess(df, "M")
                fig = fc.Theta_prediction(df, 0.8, 12)
                time.sleep(3)
                st.pyplot(fig)
        elif mode is "XGBRegressor":
            text = "This option will be available soon..."
            st.write_stream(fc.stream_data(text))
