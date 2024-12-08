# utils/data_preprocessing.py
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_input(data):
    """
    Preprocess the input data (e.g., handle encoding and scaling)
    """

    encoders = {
        'embarked' : LabelEncoder(),
        'cabin' : LabelEncoder(),
        'ticket' : LabelEncoder()
    }

    for column, encoder in encoders.items():
        if column in data:
            data[column] = encoder.fit_transform(data[column])
    
    return pd.DataFrame([data])