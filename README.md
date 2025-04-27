# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset and print the values.
3. Define X and Y array and display the value.
4. Find the value for cost and gradient.
5. Plot the decision boundary and predict the Regression value. 

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SANJITH.R
RegisterNumber: 212223230191
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
def gradient_descent(theta, X, y, alpha, num_iterations):
  m = len(y)
  for i in range(num_iterations):
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h - y) / m
    theta -= alpha * gradient
  return theta
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
  h = sigmoid(X.dot(theta))
  y_pred = np.where(h >= 0.5,1, 0)
  return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
```

## Output:
### Dataset:
![image](https://github.com/user-attachments/assets/3a3b9d78-4fed-4d40-b6c1-caf48ed8c9a4)
![image](https://github.com/user-attachments/assets/afb8bbd3-b065-4970-81d1-28e5d4abcbea)
![image](https://github.com/user-attachments/assets/12cae546-f1c1-4392-a15b-3d414fbabc51)
![image](https://github.com/user-attachments/assets/54c2d7a7-69d3-4706-9fbd-009c6de9a29d)
### Accuracy and Predicted Values:
![image](https://github.com/user-attachments/assets/60fe762e-6543-47db-9c01-603b25ff2b53)
![image](https://github.com/user-attachments/assets/d20ebb3f-227c-42e8-8b76-5ce1a8e2a650)
![image](https://github.com/user-attachments/assets/9a299ce4-5e87-4d2d-9a7a-a6ababf27471)
![image](https://github.com/user-attachments/assets/8d9d65ff-0b8a-49e3-8d3b-d083248442c0)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

