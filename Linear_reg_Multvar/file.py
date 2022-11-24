from statistics import median
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import word2number as w2n

df = pd.read_csv('H:/Data_Science/My/Linear_reg_Multvar/hiring.csv')

# Median for no values
median_testScore =math.floor(df.test_score.median())
print("Median is : ",median_testScore)

# fill NaN function for test_score ->  Data preprocessing 
df.test_score = df.test_score.fillna(median_testScore)
#print(df)

a= df.experience.el
print(a)

# reg =LinearRegression()
# reg.fit(df[['area','bedrooms','age']].values,df.price)
# m = reg.coef_
# print("Coef are ", reg.coef_)
# b=reg.intercept_
# print("intercept is : ",b)

# # Predict the prices now

# pre=reg.predict([[2800,3,20]])
# print("predict : ",pre)