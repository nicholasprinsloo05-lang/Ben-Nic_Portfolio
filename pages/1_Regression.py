import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("mpg_model.pkl")

model = load_model()

st.set_page_config(page_title="Fleet MPG Predictor", layout="wide")
st.title("Fleet Fuel Economy Predictor")

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Fleet Prediction", "Model Coefficients"])

# Tab 1: Single Vehicle Calculator
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        weight = st.slider("Vehicle Weight (lbs)", 1500, 5500, 3000)
        model_year = st.slider("Model Year", 70, 82, 76)
        displacement = st.number_input("Displacement", 50.0, 500.0, 150.0)
        horsepower = st.number_input("Horsepower", 40.0, 250.0, 100.0)
    with col2:
        acceleration = st.number_input("Acceleration", 5.0, 25.0, 15.0)
        ambient_temperature = st.number_input("Ambient Temperature (°C)", -10.0, 40.0, 20.0)
        cylinder_val = st.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
        origin_code = st.selectbox(
            "Origin", 
            [1, 2, 3], 
            format_func=lambda x: {1: "USA", 2: "Europe", 3: "Japan"}[x]
        )

    if st.button("Predict MPG", type="primary"):
        weight_c = weight - 3000
        model_year_c = model_year - 76
        weight_year_interaction = weight_c * model_year_c
        origin_map = {1: "USA", 2: "Europe", 3: "Japan"}

        input_data = pd.DataFrame({
            "displacement": [float(displacement)],
            "horsepower": [float(horsepower)],
            "acceleration": [float(acceleration)],
            "ambient_temperature": [float(ambient_temperature)],
            "weight_c": [float(weight_c)],
            "model_year_c": [float(model_year_c)],
            "weight_year_interaction": [float(weight_year_interaction)],
            "cylinders": [f"{cylinder_val}-cylinder"],
            "origin": [origin_map[origin_code]]
        })

        prediction = model.predict(input_data)[0]
        st.metric(label="Estimated Fuel Economy", value=f"{prediction:.1f} MPG")

# Tab 2: Batch CSV Fleet Processing
with tab2:
    st.subheader("Batch Prediction Pipeline")
    st.write("Upload a CSV file containing columns: `weight`, `model_year`, `displacement`, `horsepower`, `acceleration`, `ambient_temperature`, `cylinders`, `origin`.")
    
    uploaded_file = st.file_uploader("Upload Fleet CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Automatic Feature Engineering for Batch Data
        df_proc = df.copy()
        df_proc["weight_c"] = df_proc["weight"] - 3000
        df_proc["model_year_c"] = df_proc["model_year"] - 76
        df_proc["weight_year_interaction"] = df_proc["weight_c"] * df_proc["model_year_c"]
        
        # Format string categories
        origin_map = {1: "USA", 2: "Europe", 3: "Japan"}
        df_proc["cylinders"] = df_proc["cylinders"].apply(lambda x: f"{int(x)}-cylinder" if "cylinder" not in str(x) else str(x))
        df_proc["origin"] = df_proc["origin"].apply(lambda x: origin_map.get(x, str(x)))
        
        # Run inference
        feature_cols = ["displacement", "horsepower", "acceleration", "ambient_temperature", "weight_c", "model_year_c", "weight_year_interaction", "cylinders", "origin"]
        df["predicted_mpg"] = model.predict(df_proc[feature_cols]).round(2)
        
        st.dataframe(df, use_container_width=True)
        st.download_button("Download Annotated CSV", df.to_csv(index=False), "fleet_predictions.csv", "text/csv")

# Tab 3: Model Diagnostics
with tab3:
    st.subheader("Linear Regression Weights")
    regressor = model.named_steps["regressor"]
    preprocessor = model.named_steps["preprocessor"]
    
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(["cylinders", "origin"]))
    num_names = ["displacement", "horsepower", "acceleration", "ambient_temperature", "weight_c", "model_year_c", "weight_year_interaction"]
    all_features = num_names + cat_names
    
    coef_df = pd.DataFrame({
        "Feature": all_features,
        "Weight (Impact on MPG)": regressor.coef_
    }).sort_values(by="Weight (Impact on MPG)", key=abs, ascending=False)
    
    st.dataframe(coef_df, use_container_width=True)