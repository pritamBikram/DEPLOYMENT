import numpy as np
import matplotlib as plt
import pandas as pd
import pickle
dataset=pd.read_csv('datasets_192683_428563_hiring.csv')
#print(dataset.head())

dataset['experience'].fillna(0,inplace=True)
#print(dataset)
dataset['test_score(out of 10)'].fillna(dataset['test_score(out of 10)'].mean(),inplace=True)
#print(dataset)


d={
   'one':1,
   'two':2,
   'three':3,
   'four':4,
   'five':5,
   'six':6,
   'seven':7,
   'eight':8,
   'nine':9,
   'ten':10,
   'eleven':11,
   'tweleve':12
   
   
   }
dataset['experience'].replace(d,inplace=True)
#print(dataset)
#print(dataset.info())
x=dataset.drop(['salary($)'],axis=1)
y=dataset['salary($)']
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)


pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))

