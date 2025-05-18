import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import streamlit as st
import pandas as pd
import modeling

st.title("Medical Diagnosis Predictor")

model = modeling.load_model("models/best_model.joblib")
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

inputs = {}
for col in columns:
    inputs[col] = st.number_input(col, min_value=0.0)

if st.button("Predict"):
    X = pd.DataFrame([inputs])
    pred = model.predict(X)[0]
    st.write("Prediction:", "Positive" if pred == 1 else "Negative")
