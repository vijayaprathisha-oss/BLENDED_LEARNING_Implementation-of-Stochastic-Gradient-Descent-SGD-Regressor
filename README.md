# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset, remove unnecessary columns, convert categorical data to dummy variables, and separate features (X) and target (y).

2.Standardize both features and target values, then split the data into training and testing sets.

3.Train an SGD Regressor model on the training data and generate predictions on the test data. 

4.Evaluate using MSE, MAE, R², display coefficients, and plot actual vs predicted values.


## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: VIJAYAPRATHISHA J
RegisterNumber:  212225240184
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)
X=data.drop('price',axis=1)
y=data['price']
scaler = StandardScaler()
X=scaler.fit_transform(X)
y=scaler.fit_transform(np.array(y).reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)
sgd_model.fit(X_train,y_train)
y_pred=sgd_model.predict(X_test)
print('Name: VIJAYAPRATHISHA')
print('Reg No:212225240184')
print(f"{'MSE':}:{mean_squared_error(y_test,y_pred):}")
print(f"{'MAE':}:{mean_absolute_error(y_test,y_pred):}")
print(f"{'R-squared':}:{r2_score(y_test,y_pred):}")
print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)
plt.scatter(y_test,y_pred)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

```

## Output:
<img width="899" height="714" alt="image" src="https://github.com/user-attachments/assets/3f87851a-0e84-478b-8b64-8c2f95f9051d" />

<img width="639" height="637" alt="image" src="https://github.com/user-attachments/assets/4493bdf5-040d-4866-806e-0d9e2cd08ae9" />

<img width="310" height="135" alt="image" src="https://github.com/user-attachments/assets/588fc307-0e86-4130-a61d-34a25efba2ef" />

<img width="811" height="316" alt="image" src="https://github.com/user-attachments/assets/4b701043-c759-4f99-8b74-847499874ff9" />

<img width="754" height="578" alt="image" src="https://github.com/user-attachments/assets/9b75da83-4ecc-4988-96cf-1f951f8519ff" />

## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
