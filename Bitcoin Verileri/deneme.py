from statistics import linear_regression
from tracemalloc import start
import pandas as pd   
import numpy as np
import pandas_datareader as web 
import matplotlib.pyplot as plt  
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data=web.DataReader('BTC-USD',data_source='yahoo',start='2021-01-01',end='2022-09-01')
data.shape

data=data[['Close']]
data.tail()

gg=20
data['Tahmin']=data['Close'].shift(-gg)
data.tail()
print(data)

X=np.array(data.drop(['Tahmin'],1))[:-gg]
#print(X)
Y=np.array(data['Tahmin'])[:-gg]
#print(Y)


x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=25)

dtr=DecisionTreeRegressor()
lr=LinearRegression()

tree=dtr.fit(x_train,y_train)
linear=lr.fit(x_train,y_train)

xg=data.drop(['Tahmin'],1)[-gg:]
xg= xg.tail(gg)
xg=np.array(xg)
#print(xg)

dtrtahmin=dtr.predict(xg)
lintahmin=lr.predict(xg)

tahminler=lintahmin

valid=data[X.shape[0]:]
valid['Tahminler']=tahminler
plt.title("model")
plt.xlabel("günler")
plt.ylabel("fiyatlar")
plt.plot(data['Close'])
plt.plot(valid[['Close','Tahminler']])
plt.legend(['original','deger','tahmin'])
plt.show()