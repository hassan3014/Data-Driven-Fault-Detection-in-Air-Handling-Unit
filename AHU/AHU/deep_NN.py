# deep_NN.py

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from django.conf import settings
import tensorflow as tf
import os
import joblib

def train_Model():
    csv_file_path = os.path.join(settings.STATICFILES_DIRS[0], "SZCAV.csv")
    df = pd.read_csv(csv_file_path)

    feature_columns = [
        "AHU: Supply Air Temperature", "AHU: Supply Air Temperature Heating Set Point",
        "AHU: Supply Air Temperature Cooling Set Point", "AHU: Outdoor Air Temperature",
        "AHU: Return Air Temperature", "AHU: Supply Air Fan Status",
        "AHU: Supply Air Fan Speed Control Signal", "AHU: Cooling Coil Valve Control Signal",
        "AHU: Heating Coil Valve Control Signal",
        "Occupancy Mode Indicator"
    ]

    target_column = "Fault Detection Ground Truth"

    X = df[feature_columns].values
    y = df[target_column].values

    X_train = X[0:15000, 0:11]
    y_train = y[0:15000]

    X_test = X[15000:21600, 0:11]
    y_test = y[15000:21600]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(10,)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=10)

    accuracy = history.history['accuracy']
    loss = history.history['loss']


    y_pred_prob = model.predict(X_test)
    y_pred = np.round(y_pred_prob).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    model_filename = 'static/neural_network_model.joblib'
    joblib.dump(model, model_filename)

    # Calculate whether a fault is detected
    fault_detected = cm[1, 1] > 0

    return {
        'status': 'Detected' if fault_detected else 'Not Detected',
    }
