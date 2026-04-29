# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2.Import required libraries.
3.Create or load the student dataset (CGPA, IQ, Placement Status).
4.Separate input features (CGPA, IQ) and output label (Placement).
5.Split the dataset into training data and testing data.
6.Create the Logistic Regression model.
7.Train the model using training data.
8.Test the model using testing data.
9.Calculate accuracy of the model.
10.Get CGPA and IQ from the user.
11.Predict whether the student is placed or not placed.
12.Display the prediction result.
13.Stop.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
## Developed by: KAVIYA R
## RegisterNumber: 212225040179 
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = {
    'CGPA':[6.5,7.8,8.2,5.9,7.0,8.5,6.8,7.6],
    'IQ':[110,120,135,100,115,140,105,125],
    'Placement':[0,1,1,0,1,1,0,1]
}

df = pd.DataFrame(data)
X = df[['CGPA','IQ']]
y = df['Placement']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
cgpa = float(input("Enter CGPA: "))
iq = int(input("Enter IQ: "))

result = model.predict([[cgpa,iq]])

if result[0]==1:
    print("Student is Placed")
else:
    print("Student is Not Placed")
```

## Output:


<img width="963" height="188" alt="Screenshot 2026-04-29 105122" src="https://github.com/user-attachments/assets/0fb865b8-4f09-4ecc-9c18-163a2f0f25ad" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
