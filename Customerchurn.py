# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:10:14 2020

@author: ALU Student51
"""

import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,plot_confusion_matrix,plot_roc_curve,precision_score,recall_score,precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')
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
newdata['TotalCharges']=newdata['TotalCharges'].fillna(newdata['TotalCharges'].mean)
newdata['TotalCharges'] = pd.to_numeric(newdata['TotalCharges'], errors='coerce')

X=newdata[['gender','Dependents','tenure','PhoneService','InternetService','OnlineSecurity','OnlineBackup','TechSupport','PaymentMethod']]
y=newdata['Churn']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sl=StandardScaler()
X_trained= sl.fit_transform(X_train)
X_tested= sl.fit_transform(X_test)

class_name=['yes','no']
st.title('Customer churn Prediction')

st.markdown("""
Machine Learning models which predict potential customer to churn
""")
st.sidebar.title('Customer churn Prediction')

st.sidebar.markdown("""
Machine Learning models which predict potential customer to churn
""")


if st.sidebar.checkbox("show raw data",False):
    st.subheader("Customer Churn for classification")
    st.write(data)
if st.sidebar.checkbox("Show Encoded Data"):
    st.write(newdata)
if st.sidebar.checkbox("Show a Statistical Analysis"):
	st.write(newdata.describe())
    
st.sidebar.subheader('Choose Classifer')
classifier_name = st.sidebar.selectbox(
    'Choose classifier',
    ('KNN', 'SVM', 'Random Forest','Logistic Regression')
)
if classifier_name == 'SVM':
    st.sidebar.subheader('Model Hyperparmeter')
    c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='c')
    kernel= st.sidebar.radio("kernel",("linear","rbf"),key='kernel')
    gamma= st.sidebar.radio("gamma(kernel coefficiency",("scale","auto"),key='gamma')

    metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))

    if st.sidebar.button("classify",key='classify'):
        st.subheader("SVM result")
        svcclassifier= SVC(C=c,kernel=kernel,gamma=gamma)
        svcclassifier.fit(X_trained,y_train)
        y_pred= svcclassifier.predict(X_tested)
        acc= accuracy_score(y_test,y_pred)
        st.write("Accuracy:",acc.round(2))
        st.write("precision_score:",precision_score(y_test,y_pred,labels=class_name).round(2))
        st.write("recall_score:",recall_score(y_test,y_pred,labels=class_name).round(2))
        if 'confusion matrix' in metrics:
            st.subheader('confusion matrix')
            plot_confusion_matrix(svcclassifier,X_tested,y_test,display_labels=class_name)
            st.pyplot()
        if 'roc_curve' in metrics:
            st.subheader('plot_roc_curve')
            plot_roc_curve(svcclassifier,X_tested,y_test)
            st.pyplot()
        if 'precision_recall_curve' in metrics:
            st.subheader('precision_recall_curve')
            plot_roc_curve(svcclassifier,X_tested,y_test)
            st.pyplot()
        
if classifier_name == 'Logistic Regression':
    st.sidebar.subheader('Model Hyperparmeter')
    c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='r')
    max_iter= st.sidebar.slider("maximum number of iteration",100,500,key='max_item')
   

    metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))

    if st.sidebar.button("classify",key='classify'):
        st.subheader("Logistic Regression result")
        Regression= LogisticRegression(C=c,max_iter=max_iter)
        Regression.fit(X_trained,y_train)
        y_prediction= Regression.predict(X_tested)
        acc= accuracy_score(y_test,y_prediction)
        st.write("Accuracy:",acc.round(2))
        st.write("precision_score:",precision_score(y_test,y_prediction,labels=class_name).round(2))
        st.write("recall_score:",recall_score(y_test,y_prediction,labels=class_name).round(2))
        if 'confusion matrix' in metrics:
            st.subheader('confusion matrix')
            plot_confusion_matrix(Regression,X_tested,y_test,display_labels=class_name)
            st.pyplot()
        if 'roc_curve' in metrics:
            st.subheader('plot_roc_curve')
            plot_roc_curve(Regression,X_tested,y_test)
            st.pyplot()
        if 'precision_recall_curve' in metrics:
            st.subheader('precision_recall_curve')
            plot_roc_curve(Regression,X_tested,y_test)
            st.pyplot()

if classifier_name == 'Logistic Regression':
    st.sidebar.subheader('Model Hyperparmeter')
    c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='r')
    max_iter= st.sidebar.slider("maximum number of iteration",100,500,key='max_item')
   

    metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))

    if st.sidebar.button("classify",key='classify'):
        st.subheader("Logistic Regression result")
        Regression= LogisticRegression(C=c,max_iter=max_iter)
        Regression.fit(X_trained,y_train)
        y_prediction= Regression.predict(X_tested)
        acc= accuracy_score(y_test,y_prediction)
        st.write("Accuracy:",acc.round(2))
        st.write("precision_score:",precision_score(y_test,y_prediction,labels=class_name).round(2))
        st.write("recall_score:",recall_score(y_test,y_prediction,labels=class_name).round(2))
        if 'confusion matrix' in metrics:
            st.subheader('confusion matrix')
            plot_confusion_matrix(Regression,X_tested,y_test,display_labels=class_name)
            st.pyplot()
        if 'roc_curve' in metrics:
            st.subheader('plot_roc_curve')
            plot_roc_curve(Regression,X_tested,y_test)
            st.pyplot()
        if 'precision_recall_curve' in metrics:
            st.subheader('precision_recall_curve')
            plot_roc_curve(Regression,X_tested,y_test)
            st.pyplot()

if classifier_name == 'Logistic Regression':
    st.sidebar.subheader('Model Hyperparmeter')
    c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='r')
    max_iter= st.sidebar.slider("maximum number of iteration",100,500,key='max_item')
   

    metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))

    if st.sidebar.button("classify",key='classify'):
        st.subheader("Logistic Regression result")
        Regression= LogisticRegression(C=c,max_iter=max_iter)
        Regression.fit(X_trained,y_train)
        y_prediction= Regression.predict(X_tested)
        acc= accuracy_score(y_test,y_prediction)
        st.write("Accuracy:",acc.round(2))
        st.write("precision_score:",precision_score(y_test,y_prediction,labels=class_name).round(2))
        st.write("recall_score:",recall_score(y_test,y_prediction,labels=class_name).round(2))
        if 'confusion matrix' in metrics:
            st.subheader('confusion matrix')
            plot_confusion_matrix(Regression,X_tested,y_test,display_labels=class_name)
            st.pyplot()
        if 'roc_curve' in metrics:
            st.subheader('plot_roc_curve')
            plot_roc_curve(Regression,X_tested,y_test)
            st.pyplot()
        if 'precision_recall_curve' in metrics:
            st.subheader('precision_recall_curve')
            plot_roc_curve(Regression,X_tested,y_test)
            st.pyplot()
            

if classifier_name == 'Random Forest':
    st.sidebar.subheader('Model Hyperparmeter')
    n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='estimators')
    max_depth= st.sidebar.number_input("maximum depth of tree",1,20,step=1,key='max_depth')
    bootstrap= st.sidebar.radio("Boostrap sample when building trees",("True","False"),key='boostrap')


    metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))

    if st.sidebar.button("classify",key='classify'):
        st.subheader("Random Forest result")
        model= RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
        model.fit(X_trained,y_train)
        y_prediction= model.predict(X_tested)
        acc= accuracy_score(y_test,y_prediction)
        st.write("Accuracy:",acc.round(2))
        st.write("precision_score:",precision_score(y_test,y_prediction,labels=class_name).round(2))
        st.write("recall_score:",recall_score(y_test,y_prediction,labels=class_name).round(2))
        if 'confusion matrix' in metrics:
            st.subheader('confusion matrix')
            plot_confusion_matrix(model,X_tested,y_test,display_labels=class_name)
            st.pyplot()
        if 'roc_curve' in metrics:
            st.subheader('plot_roc_curve')
            plot_roc_curve(model,X_tested,y_test)
            st.pyplot()
        if 'precision_recall_curve' in metrics:
            st.subheader('precision_recall_curve')
            plot_roc_curve(model,X_tested,y_test)
            st.pyplot()            
        