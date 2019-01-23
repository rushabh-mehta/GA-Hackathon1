# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 23:40:54 2018

@author: rusha
"""

import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv("diamonds.csv")
print(list(df))
df = df.iloc[:,5:]
print(df.head())
y=df["price"]
X=df[["depth","table","y","z"]]
random_state=9
test_size=0.2
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=9,test_size=0.33)
model = LinearRegression()
model.fit(X_train,y_train)
y_predict= model.predict(X_test)
residual = y_test-y_predict
#plt.scatter(y_predict,residual)
#stats.probplot(residual,plot=plt)
print(mean_absolute_error(y_predict,y_test))
print(mean_squared_error(y_predict,y_test))
print(r2_score(y_test,y_predict))
plt.show()
