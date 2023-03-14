import numpy as np
import pandas as pd
from keras.models import load_model


def detect_disease(filename):
    dataframe = pd.read_csv(filename)

    dataframe = dataframe.astype('float32')
    dataframe = dataframe.to_numpy()

    testing_features = dataframe[:, 1:]

    best_model = load_model('Assets/Model/cnn_model.h5')
    prediction = best_model.predict(testing_features)

    if prediction[0] < 0.5:
        return "negative"
    elif prediction[0] >= 0.5 and prediction[0] <= 1:
        return "positive"

    return "false"


"""detect_disease("Assets/csv_data/patient.csv") """
