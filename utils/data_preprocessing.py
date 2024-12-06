# utils/data_preprocessing.py
import pandas as pd

def preprocess_input(data):
    df = pd.DataFrame(data)
    df = pd.get_dummies(df)  # Sesuaikan dengan preprocessing yang sama dengan data training
    return df
