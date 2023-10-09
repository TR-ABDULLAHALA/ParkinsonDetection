import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Parkinson veri setini okutma
df=pd.read_csv('parkinsons.data')

#Veri setinin ilk beş satırını göster
head=df.head()
print(head)

#Veri setindeki 'status' sütununu hedef değişken olarak ayırma
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values

#Etiketlerin sayısını görüntüleme
print(labels[labels==1].shape[0], labels[labels==0].shape[0])

#Veri özelliklerini ölçeklendirme
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

#Verileri eğitim ve test setlerine ayırma
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

#XGBoost sınıflandırıcı modelini oluşturma ve eğitme
model=XGBClassifier()
model.fit(x_train,y_train)

#Test seti üzerinde modelin performansı ve doğruluk oranı
y_pred=model.predict(x_test)
print("Doğruluk Oranı:",accuracy_score(y_test, y_pred)*100)