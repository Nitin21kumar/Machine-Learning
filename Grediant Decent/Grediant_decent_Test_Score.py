# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:22:43 2024

@author: Ankit Sharma
"""
import numpy as np
import pandas as pd

data = pd.read_csv("test_scores.csv")

data.head()

x = data.iloc[:,1]

x

y = data.iloc[:,-1]


m_curr = b_curr = 0

iteration = 10000

learning_rate = 0.0002

n = len(x)

print(n)
print(x,y)
for i in range(iteration):
    
    y_pred = m_curr * x + b_curr
    
    #cost = (1/n)*sum([val**2 for val in (y - y_pred)])
    
    partialDerivative_m = -(2/n)*sum(x*(y - y_pred))
    
    partialDerivative_b = -(2/n)*sum(y - y_pred)
    
    m_curr = m_curr - learning_rate*partialDerivative_m
    b_curr = b_curr - learning_rate*partialDerivative_b
    
    print(m_curr, b_curr)
    
    
print(m_curr, b_curr)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    