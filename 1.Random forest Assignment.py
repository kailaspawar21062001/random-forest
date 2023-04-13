# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 20:10:39 2023

@author: kailas
"""

################################################################################
1]PROBLEM:  'Company_Data.csv'


    
BUSINESS OBJECTIVE:-    
Approach - A Random Forest can be built with target variable Sale (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  




#Import Liabrary
import pandas as pd
import numpy as np 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Dataset
data=pd.read_csv("D:/data science assignment/Assignments/15.random forest/Company_Data (1).csv")

#EDA
data.head()
data.tail()
data.shape
data.info()
data.describe()
data.isna().sum()#To check NA Values.


#Data Preprocessing
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
data['ShelveLoc']=l.fit_transform(data['ShelveLoc'])
data['Urban']=l.fit_transform(data['Urban'])
data['US']=l.fit_transform(data['US'])

#Discritization for target variable(Sales)
data['Salaries_nefw'] = pd.cut(data['Sales'], bins=[min(data.Sales) - 1, 
                                                  data.Sales.mean(), max(data.Sales)], labels=["Low","High"])

data.describe()
data.head()
data.drop(['Sales'],axis=1,inplace=True)

inpu=data.iloc[:,0:10]#Predictors
target=data.iloc[:,[10]]#Target

#Split the Dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inpu,target,test_size=0.2)


#Model Buliding
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=500,n_jobs=1,random_state=42)
#I tune/Prunning the model again and again and i found (n_estimators=500) is the optimum number for achieving the good accuracy.

model.fit(x_train,y_train)

#Evaluations On Train data
trainpred=model.predict(x_train)
accuracy_score(y_train,trainpred)
train_report=classification_report(y_train,trainpred)
confusion_matrix(y_train,trainpred)

#Evaluations On Test data
testpred=model.predict(x_test)
accuracy_score(y_test,testpred)
test_report=classification_report(y_test,testpred)
confusion_matrix(y_test,testpred)

#################################################################################
2]PROBLEM ::'Fraud_check.csv'

BUSINESS OBJECTIVE:-Use Random Forest to prepare a model on fraud data. 

Note=(Treating those who have taxable_income <= 30000 as "Risky" and others are "Good")


data=pd.read_csv("D:/data science assignment/Assignments/15.random forest/Fraud_check (1).csv")

#EDA
data.describe()
data.head()
data.tail()
data.shape
data.info()
data.isna().sum()

#Rename 
data=data.rename(columns={'Taxable.Income':'tax'})
data.info()


#Creating Discritization..>=3000 as 'Risky' and remainings are good.
ma=['Risky','Good']
bi=[0,30000,99619]
data['tax_new']=pd.cut(x=data.tax,bins=bi,labels=ma,retbins=False,duplicates='raise',ordered=True)
data.head()

#Dropping the original 'tax' feature,which is continuous in nature.
data.drop(['tax'],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
data['Undergrad']=l.fit_transform(data['Undergrad'])
data['Marital.Status']=l.fit_transform(data['Marital.Status'])
data['Urban']=l.fit_transform(data['Urban'])


inpu=data.iloc[:,0:5]#Predictors
target=data.iloc[:,[5]]#Target

#Split the Dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inpu,target,test_size=0.2)



#Model Building
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=500,n_jobs=1)
#I tune/Prunning the model again and again and i found (n_estimators=500) is the optimum number for achieving the good accuracy.

model.fit(x_train,y_train)


#Evalutions on Train Data
trainpred=model.predict(x_train)
accuracy_score(y_train,trainpred)
train_report=classification_report(y_train,trainpred)
confusion_matrix(y_train,trainpred)


#Evalutions on Train Data
testpred=model.predict(x_test)
accuracy_score(y_test,testpred)
test_report=classification_report(y_test,testpred)
confusion_matrix(y_test,testpred)
