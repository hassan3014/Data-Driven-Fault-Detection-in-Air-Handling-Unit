from django.shortcuts import render,redirect,HttpResponse
from work.models import Register,value
from AHU import impFunc
import pandas as pd
from django.http import JsonResponse
import os, base64
from .knn import knn
from .deep_NN import train_Model
from .SVM import svm



def index(request):
    return render(request,'index.html')


def login(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        u = Register.objects.get(username=username)
        if u.password == password:
            user = request.session.get('USER')
            user = {
                'username': u.username,
            }
            request.session['USER'] = user
            request.session.save()
            return redirect('dashboard')
        else:
            data = {
                "alert":True,
            }
            return render(request,'index.html',data)
        
    else:
        return redirect('home')



# def dash(request, sec=None):
#     first_record = value.objects.last()

#     if first_record:
#         column_names = [field.name for field in first_record._meta.fields]

#         excluded_fields = ['id']

#         column_names = [col for col in column_names if col not in excluded_fields]

#         column_data = []

#         for col in column_names:
#             if col == 'supply_air_fan_status' or col == 'occupancy_mode' or col == 'fault_detection':
#                 column_data.append(0)
#                 print('true')
#                 # column_data.append(int.from_bytes(b'\x01\x00\x00\x00\x00\x00\x00\x00', byteorder='little'))
#             else:
#                 column_data.append(getattr(first_record, col))
#             print(col , " - " , getattr(first_record, col),'\n')
#         print(column_data)
#     else:
#         column_names = []
#         column_data = []

#     idx = [f'mychart{i}' for i in range(1, 16)]

#     data = {
#         "labels": column_names[1:],
#         "values": column_data[1:],
#         # "date": dates,
#         # "time": times,
#         'idx':idx,
#     }

#     return render(request, 'dashboard.html', {"data": data})



#     column_names = [field.name for field in value._meta.fields]

#     response_text = "Column Names:\n" + "\n".join(column_names)

#     return HttpResponse(response_text)
# ==================================================================================
# ==================================================================================
# ==================================================================================
# ==================================================================================
# ==================================================================================
# views.py


from django.shortcuts import render
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

def train_svm_model():
    data = pd.read_csv('static/SZCAV.csv')  # Replace with the actual file path
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

    # Calculate whether a fault is detected
    fault_detected = cm[1, 1] > 0

    return {
        'fault_detected': fault_detected,
    }

# def dash(request, sec=None):
#     # Existing code ...
#
#     # Assuming train_svm_model() returns the result in the 'fault_detected' key
#     svm_result = train_svm_model().get('fault_detected', False)
#
#     data = {
#         "labels": column_names.tolist(),
#         "values": column_data,
#         'idx': idx,
#         'nn_result': nn_result,
#         'svm_result': svm_result,  # Pass the SVM result to the template
#     }
#
#     return render(request, 'dashboard.html', {"data": data})


def dash(request, sec=None):
    df = pd.read_csv('static/SZCAV.csv')
    df = df.sample(n=1)

    column_names = df.columns
    column_names = column_names[1:15]
    column_data = []

    for col in column_names:
        column_data.append(df[col].tolist())


    idx = [f'mychart{i}' for i in range(1, 15)]

    if sec is not None:
        average = impFunc.timebase(sec)
        column_data = average

    # Assuming train_Model() returns the result in the 'fault_detected' key
    # nn_result = train_Model()['fault_detected']
    nn_result = train_Model().get('status', 'Not Detected')
    svm_result = svm().get('status', 'Not Detected')
    knn_result = knn().get('status', 'Not Detected')


    data = {
        "labels": column_names.tolist(),
        "values": column_data,
        'idx': idx,
        'nn_result': nn_result,
        'svm_result': svm_result,
        'knn_result': knn_result,
    }

    return render(request, 'dashboard.html', {"data": data})




def logout(request):
    try:
        n = request.session.pop('USER')
        print("user popped: ", n['username'])
        request.session.save()
    
        return redirect('home')
    except:
        return redirect('home')


