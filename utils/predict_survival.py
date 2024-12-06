import pickle

def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def make_prediction(data, model):
    return model.predict(data) [0]