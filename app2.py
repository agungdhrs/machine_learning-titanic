import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data, encoders, features):
    # Preprocess the data using the provided encoders and features
    for col in ['ticket', 'cabin', 'embarked']:
        data[col] = data[col].astype(str)
        if col in encoders:
            data[col] = data[col].apply(
                lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else np.nan
            )
    return data[features]

def load_model_and_encoders():
    # Load the trained model and encoders from the saved files
    model = joblib.load('model/random_forest_model.pkl')
    encoders = joblib.load('utils/encoders.pkl')
    return model, encoders

st.title("Titanic Survival")

uploaded_file = st.file_uploader("Upload file Excel", type=["xls", "xlsx"])
if uploaded_file:
    try:
        # Load the uploaded file into a pandas DataFrame
        data = pd.read_excel(uploaded_file)
        st.write("Data berhasil dimuat:")
        st.write(data.head())
        # st.write("Kolom dalam data:", data.columns.tolist())

        # Load model and encoders
        model, encoders = load_model_and_encoders()

        # Preprocess data
        features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'ticket', 'cabin', 'embarked']
        processed_data = preprocess_data(data, encoders, features)

        if st.button("Prediksi"):
            # Predict the outcomes
            predictions = model.predict(processed_data)
            data['survive'] = ["Survive" if p == 1 else "Not Survive" for p in predictions]

            # Display results
            if 'name' in data.columns and 'survive' in data.columns:
                st.write("Hasil Prediksi: ")
                st.write(data[['name', 'survive']])
            else:
                st.write("Kolom 'name' atau 'survive' tidak ditemukan. Berikut hasil prediksi:")
                st.write(data[['survive']])
    except Exception as e:
        st.error(f"Kesalahan: {e}")