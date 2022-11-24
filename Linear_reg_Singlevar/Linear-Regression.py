from pyexpat import model
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('H:/Data_Science/My/Linear_reg_Singlevar/canada.csv')

print(df.head())

reg= LinearRegression()
reg.fit(df[['year']].values,df.income)

reg.predict([[2020]])


# m = reg.coef_c
# x=2020
# print("slope or value of m :" , m)
# # intercept
# b = reg.intercept_
# print("Intercept is : ", b)

# y= (m*x) + b

# print("value of y of the value we want to predict is : ",y)


plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(df.year,df.income,color='red',marker='+')
plt.plot(df.year,reg.predict(df[['year']].values),color='blue')
plt.show()

import pickle
# with open('model_pickle','wb') as f:
#     pickle.dump(reg,f)

with open('model_pickle','rb') as f:
    mp=pickle.load(f)

mp.predict(2020)