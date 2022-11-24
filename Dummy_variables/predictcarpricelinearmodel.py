# -*- coding: utf-8 -*-
"""predictCarPriceLinearModel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xPeecp_QzJI1aEYVt8FclGc2WFq-VlnS
"""

import pandas as pd
df = pd.read_csv("carprices.csv")
df

duplicate = pd.get_dummies(df['Car Model'])
duplicate

merge = pd.concat([df,duplicate],axis='columns')
merge

final= merge.drop(['Car Model','Mercedez Benz C class'],axis='columns')
final

X=final.drop(['Sell Price($)'],axis='columns').values
X

y=final['Sell Price($)'].values
y

from sklearn.linear_model import LinearRegression
frame = LinearRegression()

frame.fit(X,y)

frame.score(X,y)

frame.predict([[45000,4,0,0]])

frame.predict([[86000,7,0,1]])