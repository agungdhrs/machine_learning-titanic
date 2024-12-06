from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import os

# Data contoh untuk encoder (sesuaikan dengan data asli Anda)
data = {
    'ticket': ['A/5', 'PC', 'STON', '113803'],
    'cabin': ['C85', 'C123', 'E46', 'B96 B98'],
    'embarked': ['S', 'C', 'Q']
}

df = pd.DataFrame(data)

# Inisialisasi LabelEncoder untuk setiap kolom
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])  # Fit-transform data
    encoders[column] = le  # Simpan encoder untuk kolom

# Simpan encoder ke file utils/encoder.pkl
output_path = "utils/encoder.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'wb') as file:
    pickle.dump(encoders, file)
