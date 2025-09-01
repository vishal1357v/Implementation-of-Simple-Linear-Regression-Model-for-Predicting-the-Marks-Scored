# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored


## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook


## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.



## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vishal P
RegisterNumber:  212224230306
*/
```

```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:

### Head Values
<img width="1090" alt="Screenshot 2024-04-05 at 9 36 33 AM" src="https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/4ddf5d62-c261-42be-8b67-6f6df35f3d36">


### Tail Values
<img width="1090" alt="Screenshot 2024-04-05 at 9 37 06 AM" src="https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/dfa9fd82-e723-4ed4-aaec-ad66ffa348e1">



### Compare Dataset
<img width="1090" alt="Screenshot 2024-04-05 at 9 37 18 AM" src="https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/9118849a-2323-4b9c-a362-3dc8d060587a">


### Predication values of X and Y
<img width="1090" alt="Screenshot 2024-04-05 at 9 37 31 AM" src="https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/5ed7921d-08e1-408e-8383-6899933b01ee">


### Training set
<img width="1090" alt="Screenshot 2024-04-05 at 9 37 44 AM" src="https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/9fae6c7e-e00d-449f-8f16-cbbe94095e76">


### Testing Set
<img width="1090" alt="Screenshot 2024-04-05 at 9 37 50 AM" src="https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/bea6f3d4-6bf1-4c28-b7ff-167a07ec1d30">

### MSE,MAE and RMSE
<img width="1090" alt="Screenshot 2024-04-05 at 9 38 02 AM" src="https://github.com/gauthamkrishna7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141175025/0f1f6538-a2a8-4c7c-838e-8ccbca0c7b6c">

<br>
<br>
<br>
<br>
<br>

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
