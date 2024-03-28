import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from keras.models import load_model
import joblib


# inp = np.array([[50.3, 20.50, 20.0, 35.5, 40.0, 50.0, 80.4, 30.4, 22.0, 40.50, 90.0]])

# def pred(data):
#     model = load_model('AHU\\model.h5')

#     y_pred_prob = model.predict(data)
#     return int(y_pred_prob)
# def pred1(data):
#     model = load_model('AHU\\knn_model.joblib')

#     y_pred_prob = model.predict(data)
#     return int(y_pred_prob)
# def pred2(data):
#     model = load_model('AHU\\neural_network_model.joblib')

#     y_pred_prob = model.predict(data)
#     return int(y_pred_prob)
# def pred3(data):
#     model = load_model('AHU\svm_model.joblib')

#     y_pred_prob = model.predict(data)
#     return int(y_pred_prob)

# def pred1(data):
#     model = joblib.load('AHU/knn_model.joblib')
#     y_pred_prob = model.predict(data)
#     return int(np.argmax(y_pred_prob))

# def pred2(data):
#     model = joblib.load('AHU/neural_network_model.joblib')
#     y_pred_prob = model.predict(data)
#     return int(np.argmax(y_pred_prob))

# def pred3(data):
#     model = joblib.load('AHU/svm_model.joblib')
#     y_pred_prob = model.predict(data)
#     return int(np.argmax(y_pred_prob))


def timebase(val):
    a = int(val)
    df = pd.read_csv('static\SZCAV.csv').tail(a)
    column_names = df.columns
    
    column_names = column_names[1:16]
    avrg = []
    for col in column_names:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        column_average = df[col].tail(val).mean()
        avrg.append(column_average)

    return avrg


