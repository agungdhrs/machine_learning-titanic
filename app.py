import streamlit as st
import pandas as pd
import numpy as np
import pickle
from utils.predict_survive import load_model

model = load_model()
encoders = pickle.load(open('utils/encoders.pkl', 'rb'))
features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else ['pclass', 'age', 'sibsp', 'parch', 'fare', 'ticket', 'cabin', 'embarked']

st.title("Titanic Survival")

nama = st.text_input("Nama")
data = {
    "pclass": st.selectbox("Kelas Kapal", [1, 2, 3]),
    "age": st.number_input("Umur", min_value=0, max_value=100),
    "sibsp": st.number_input("Jumlah Saudara", min_value=0, max_value=100),
    "parch": st.number_input("Kerabat", min_value=0, max_value=100),
    "fare": st.number_input("Biaya Tiket", min_value=0, max_value=100),
    "ticket": st.text_input("Nomor Tiket"),
    "cabin": st.text_input("Kabin"),
    "embarked": st.selectbox("Pelabuhan", ['C', 'Q', 'S'])
}

def transform_data(data, encoders, features):
    for k, v in data.items():
        if k in encoders and v not in encoders[k].classes_:
            encoders[k].classes_ = np.append(encoders[k].classes_, v)
        data[k] = encoders[k].transform([v])[0] if k in encoders else v
    return pd.DataFrame([data])[features]

if st.button("Proses"):
    try:
        if not nama.strip():
            st.error("Nama tidak boleh kosong")
        else:
            input_data = transform_data(data, encoders, features)
            result = model.predict(input_data.values)
            outcome = "Tewas" if result[0] == 0 else "Selamat"
            st.success(f"Hasil: {nama} {outcome}")
    except Exception as e:
        st.error(f"Kesalahan: {e}")