import streamlit as st

st.set_page_config(page_title="ML Portal", layout="centered")
st.title("Machine Learning Portal")
st.write("Welcome. Select a predictive model from the sidebar to begin.")

st.page_link("pages/1_Regression.py", label="Fleet MPG Predictor (Regression)", icon="🚗")
st.page_link("pages/2_Classification.py", label="Categorical Predictor (Classification)", icon="📊")