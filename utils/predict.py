def make_prediction(model, data):
    prediction = model.predict(data)
    return prediction[0]