import streamlit as st
from pathlib import Path

temporal = Path("data/to_predict")


st.title("Energy Predictor")
st.markdown("Make your predictions here!")

st.divider()

with st.sidebar:
    if st.button("clear"):
        items = list(temporal.iterdir())
        for item in items:
            try:
                item.unlink()
            except Exception as e:
                st.error(f"There's an error: {e}")


file_uploader = st.file_uploader(
    "Upload the CSV file that contais the energy information", type=["csv"]
)
if file_uploader is not None:
    with open(temporal / "temporal.csv", "wb") as f:
        f.write(file_uploader.getvalue())


# Session state
if "uploaded_csv" not in st.session_state:
    st.session_state.uploaded_csv = None
