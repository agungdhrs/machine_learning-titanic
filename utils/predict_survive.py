import joblib
from utils.data_preprocessing import preprocess_input

def load_model():
    """
    Load the model from the saved file.
    """
    model = joblib.load('model/random_forest_model.pkl')
    return model

def make_prediction(data, model):
    """
    Use the model to make predictions on the input data.
    """
    preprocess_data = preprocess_input(data)
    prediction = model.predict(preprocess_data)
    return prediction[0]