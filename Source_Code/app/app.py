import streamlit as st
import pandas as pd
import joblib

st.title("Market Price Prediction")

model = joblib.load("models/final_model.pkl")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    preds = model.predict(df)
    st.write("Predicted Close Price:", preds[-1])
