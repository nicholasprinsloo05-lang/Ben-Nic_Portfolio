import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained model (unwraps GridSearchCV automatically if needed)
@st.cache_resource
def load_model():
    loaded = joblib.load('bean_model.pkl')
    if hasattr(loaded, 'best_estimator_'):
        return loaded.best_estimator_
    return loaded

model = load_model()

st.title("Agricultural Sorting: Dry Bean Classifier")
st.write("Optimize crop sorting and eliminate batch contamination using Neural Networks.")

# Target variety names aligned with LabelEncoder alphabetical order
CLASS_NAMES = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

st.sidebar.header("Bean Morphological Parameters")

# Primary user inputs
area = st.sidebar.number_input("Area (px)", min_value=10000, max_value=250000, value=50000)
perimeter = st.sidebar.number_input("Perimeter", min_value=100.0, max_value=2000.0, value=900.0)
compactness = st.sidebar.slider("Compactness", min_value=0.40, max_value=1.00, value=0.80, step=0.01)
solidity = st.sidebar.slider("Solidity", min_value=0.80, max_value=1.00, value=0.98, step=0.01)

if st.button("Classify Bean Variety"):
    # 1. Retrieve expected feature names from fitted model pipeline
    if hasattr(model, "feature_names_in_"):
        required_features = list(model.feature_names_in_)
    else:
        required_features = [
            "area", "perimeter", "major_axis_length", "minor_axis_length", 
            "aspect_ratio", "eccentricity", "convex_area", "equivalent_diameter", 
            "extent", "solidity", "roundness", "compactness", 
            "shape_factor_1", "shape_factor_2", "shape_factor_3", "shape_factor_4"
        ]

    # 2. Map inputs using exact snake_case feature names
    default_values = {
        "area": float(area),
        "perimeter": float(perimeter),
        "major_axis_length": 320.0,
        "minor_axis_length": 200.0,
        "aspect_ratio": 1.50,
        "eccentricity": 0.75,
        "convex_area": float(area * 1.03),
        "equivalent_diameter": 250.0,
        "extent": 0.75,
        "solidity": float(solidity),
        "roundness": 0.85,
        "compactness": float(compactness),
        "shape_factor_1": 0.006,
        "shape_factor_2": 0.001,
        "shape_factor_3": 0.65,
        "shape_factor_4": 0.99
    }

    # 3. Construct DataFrame matching exact feature names
    input_dict = {col: [float(default_values.get(col, 0.0))] for col in required_features}
    input_data = pd.DataFrame(input_dict)

    # 4. Predict and handle both integer indices and string labels gracefully
    prediction_raw = model.predict(input_data)[0]

    if isinstance(prediction_raw, (int, np.integer)):
        predicted_label = CLASS_NAMES[prediction_raw] if prediction_raw < len(CLASS_NAMES) else f"Class {prediction_raw}"
    else:
        predicted_label = str(prediction_raw).upper()

    st.success(f"**Predicted Variety:** {predicted_label}")