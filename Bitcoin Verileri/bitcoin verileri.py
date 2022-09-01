from datetime import datetime
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader as web
from sklearn.model_selection import train_test_split
import time



#btc_data = web.get_data_yahoo(['BTC-USD'],start=datetime.datetime(2018, 1, 1), end=datetime.datetime(2022, 08, 29))['Close']


data=yf.download("BTC-USD",start="2018-01-01",end="2022-09-01")
y=data.Close
data.drop(['Close'],axis=1,inplace=True)
data.drop(['Adj Close'],axis=1,inplace=True)

xtrain,xtest,ytrain,ytest=train_test_split(data,y,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(xtrain, ytrain)

tahmin=lr.predict(xtest)

plt.plot(y)
plt.show()
"""
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest, tahmin)
print(cm)"""

