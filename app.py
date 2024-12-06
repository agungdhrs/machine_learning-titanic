import streamlit as st
import pandas as pd
import numpy as np
import pickle
from utils.predict import load_model, make_prediction

# Muat model prediksi dan encoder
model = load_model()
with open('utils/encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

st.title("Prediksi Keselamatan Penumpang Kapal")

# Input Form
nama = st.text_input("Nama")
pclass = st.selectbox("Kelas Penumpang (Pclass)", [1, 2, 3])
age = st.number_input("Umur (Age)", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Jumlah Saudara/Istri/Anak (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Jumlah Orang Tua/Anak (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Tarif Tiket (Fare)", min_value=0.0, max_value=500.0, value=50.0)
ticket = st.text_input("Nomor Tiket (Ticket)")
cabin = st.text_input("Kabina (Cabin)")
embarked = st.selectbox("Pelabuhan Naik (Embarked)", ['C', 'Q', 'S'])

# Data dictionary
data = {
    "pclass": pclass,
    "age": age,
    "sibsp": sibsp,
    "parch": parch,
    "fare": fare,
    "ticket": ticket,
    "cabin": cabin,
    "embarked": embarked
}

# Fungsi transformasi data
def transform_data(data, encoders):
    transformed_data = {}
    for key, value in data.items():
        if key in encoders:  # Jika kolom memiliki encoder
            if value not in encoders[key].classes_:
                # Tambahkan nilai baru sementara ke classes_
                encoders[key].classes_ = np.append(encoders[key].classes_, value)
            
            # Transformasikan nilai ke bentuk numerik
            transformed_data[key] = encoders[key].transform([value])[0]
        else:
            # Jika tidak ada encoder, gunakan nilai asli
            transformed_data[key] = value
    return pd.DataFrame([transformed_data])

# Tombol prediksi
if st.button("Proses Prediksi"):
    try:
        # Transformasikan data menggunakan encoder
        input_data = transform_data(data, encoders)
        
        # Prediksi hasil
        result = make_prediction(input_data, model)
        outcome = "Survive" if result == 1 else "Tidak Survive"
        
        st.write(f"Hasil Prediksi: **{outcome}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
