import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score

st.title('Streamlit Example')
data=pd.read_csv('Customer-Churn.csv')
print(data.head())
print(data.info())
print(data.describe())

#fetuare extracting
#q1 Using the given dataset extract the relevant features that can define a customer churn. [5]
cols=['gender','Dependents','tenure','PhoneService','InternetService','OnlineSecurity','OnlineBackup','TechSupport','PaymentMethod','TotalCharges','Churn']
newdata=data[cols]
print(newdata.info())

le=LabelEncoder()
newdata['gender']=le.fit_transform(newdata['gender'])
newdata['Dependents']=le.fit_transform(newdata['Dependents'])
newdata['PhoneService']=le.fit_transform(newdata['PhoneService'])
newdata['InternetService']=le.fit_transform(newdata['InternetService'])
newdata['OnlineSecurity']=le.fit_transform(newdata['OnlineSecurity'])
newdata['OnlineBackup']=le.fit_transform(newdata['OnlineBackup'])
newdata['TechSupport']=le.fit_transform(newdata['TechSupport'])
newdata['PaymentMethod']=le.fit_transform(newdata['PaymentMethod'])
newdata['Churn']=le.fit_transform(newdata['Churn'])
newdata.isnull().sum()
X=newdata[['gender','Dependents','tenure','PhoneService','InternetService','OnlineSecurity','OnlineBackup','TechSupport','PaymentMethod','TotalCharges']]
y=newdata['Churn']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

random= RandomForestClassifier(n_estimators=100)
random.fit(X_train,y_train)
y_pred= random.predict(X_test)

print(accuracy_score(y_test,y_pred))











