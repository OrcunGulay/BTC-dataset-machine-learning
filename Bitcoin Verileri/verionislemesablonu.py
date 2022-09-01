# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('BTC-Daily.csv')
#pd.read_csv("veriler.csv")
#test
#print(veriler)

tarih =pd.DataFrame(veriler.iloc[:,1:2].values)  #bağımsız değişkenler
fiyatlar=pd.DataFrame(veriler.iloc[:,3:6].values)
voltalite= pd.DataFrame(veriler.iloc[:,8:].values)
x1=pd.concat([tarih,fiyatlar],axis=1)
x=pd.concat([fiyatlar,voltalite],axis=1)

y = veriler.iloc[:,6:7].values #bağımlı değişken
#print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
"""
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)
"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
pol=PolynomialFeatures(degree=4)

sonuc=pol.fit_transform(x.values)
reg=LinearRegression()
kkk=reg.fit(sonuc,y)

print (kkk)

















