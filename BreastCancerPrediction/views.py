from django.shortcuts import render;
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def home(request):
    return render(request,"home.html")
def predict(request):
    return render(request,"predict.html")
def result(request):
    data = pd.read_csv('cancer.csv')
    data = data.drop(['id','Unnamed: 32'],axis=1)
    X=data.drop('diagnosis',axis=1)
    y=data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    user_input = []
    for i in range(3, 33):  # Assuming your input fields are named 'n3', 'n4', ..., 'n32'
        try:
            value = float(request.GET[f'n{i}'])
        except ValueError:
            return render(request, "predict.html", {"error_message": "Invalid input! Please enter numeric values."})

        user_input.append(value)

    # Make predictions using the trained model
    prediction = classifier.predict(np.array(user_input).reshape(1, -1))
    pred = prediction[0]
    if pred == 'M':
        result="The Prediction result is : Malignant"
    else:
        result="The Prediction result is :Benign "
    return render(request, "predict.html", {"result2": result})