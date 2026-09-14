import streamlit as st

st.set_page_config(page_title="BEN-NIC ML Portfolio", page_icon="⚙️", layout="wide")
st.title("Machine Learning Portal")
with st.expander("About This Portfolio"):
    st.write("This application demonstrates end-to-end machine learning deployment. It features a regression model for predicting vehicle fuel efficiency and a classifier for agricultural sorting. Both pipelines include automated feature engineering and robust input validation.")
st.write("Welcome. Select a predictive model from the sidebar to begin.")

st.page_link("pages/1_Regression.py", label="Fleet MPG Predictor (Regression)", icon="🚗")
st.page_link("pages/2_Classification.py", label="Categorical Predictor (Classification)", icon="📊")
