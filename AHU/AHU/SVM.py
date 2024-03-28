import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from django.conf import settings
import os
import joblib


def svm():
    csv_file_path = os.path.join(settings.STATICFILES_DIRS[0], "SZCAV.csv")
    data = pd.read_csv(csv_file_path)
    data = data.drop(labels=["Datetime"], axis=1)
    data.replace('#VALUE!', 0, inplace=True)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    model_filename = 'static/svm_model.joblib'
    joblib.dump(svm_classifier, model_filename)

    # Calculate whether a fault is detected
    fault_detected = cm[1, 1] > 0

    return {
        'status': 'Detected' if fault_detected else 'Not Detected',
    }
