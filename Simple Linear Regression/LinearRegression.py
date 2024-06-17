# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:46:50 2024

@author: Ankit Sharma
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv('mycar.csv')

data.head(10)

X = data.iloc[:,:-1].values

Y = data.iloc[:,1].values

plt.scatter(X,Y)

plt.show()

data.corr()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)

from sklearn.linear_model import LinearRegression

myModel=LinearRegression()

myModel.fit(x_train,y_train)

myModel.coef_

myModel.intercept_

y_pred=myModel.predict(x_test)

y_pred






