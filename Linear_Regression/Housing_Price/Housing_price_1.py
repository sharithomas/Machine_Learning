# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:23:06 2020

@author: windows
"""

#DataSet: Housing Price
#remove Nan,Remove outlier , Label Encoding
import pandas as pd
import numpy as np
import seaborn as sns #visulaization
import matplotlib.pyplot as plt  #visulaization

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,r2_score 

sns.set(color_codes=True)
df=pd.read_csv("Housing_Price_170720_test.csv")
df.columns
df.dtypes

df.isnull().mean()*100
df.shape

#drop Nan
df=df.dropna()
df.shape

#handling outliers
sns.boxplot(x=df['Width'])
sns.boxplot(x=df['Length'])
sns.boxplot(x=df['Floor'])
sns.boxplot(x=df['Balcony_Area'])

#removing outliers
Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1
print(IQR)

df=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR))).any(axis=1)]
df.shape

sns.boxplot(x=df["Length"])
sns.boxplot(x=df["Width"])

df['Area']=df['Length']*df['Width']
df=df.drop(['Length','Width'],axis=1)
df.shape

#HANDLING CATEGORICAL VARIABLES- LABEL ENCODING       
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Special_Feature']=le.fit_transform(df['Special_Feature'])

df=df.drop(['Id'],axis=1)

df.shape

#train test split
from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train,df_test=train_test_split(df,train_size=0.7,test_size=0.3,random_state=100)
#random state should set as 100 or any number else it will pick different data set each time
y_train=df_train.pop('Price') #target pop from data frame train 
x_train=df_train # after popping remianing value set as x_train

#method1 statsmodel
import statsmodels.api as sm

#add constant- in sm model look like 1*m1+2*m2+...+0, in sm model constant c would be zero to add a constant explicitly 
#here we are using add_constant()
x_train_lm=sm.add_constant(x_train)
    
#create first fitted model
lr=sm.OLS(y_train,x_train_lm).fit() #OLS- ordinary list square
lr.params
lr.summary()
#analysis of results
#1)f-stati=mssb/mssw , f should big as possible so that mssb should high(more variance) and mssw should low
#f-ststi>20 consider as good

#2) probability of f-stati<0.05 or near to zero is consider as confident model
y_test=df_test.pop('Price')
x_test=df_test

x_test_lm=sm.add_constant(x_test) # in sm model y=1*m1+2*m2+...+0, to add a constant 
#instead of zero using this

y_pred_lm=lr.predict(x_test_lm)
r2_test=r2_score(y_test,y_pred_lm)

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = x_train_lm.columns
vif['VIF'] = [variance_inflation_factor(x_train_lm.values, i) for i in range(x_train_lm.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

x_train_lm_1=x_train_lm.drop(['Balcony_Area'],axis=1)
lr_1=sm.OLS(y_train,x_train_lm_1).fit()
lr_1.params
lr_1.summary()

x_test_lm_1=x_test_lm.drop(['Balcony_Area'],axis=1)
y_pred_lm_1=lr_1.predict(x_test_lm_1)
r2_test_1=r2_score(y_test,y_pred_lm_1)

vif = pd.DataFrame()
vif['Features'] = x_train_lm_1.columns
vif['VIF'] = [variance_inflation_factor(x_train_lm_1.values, i) for i in range(x_train_lm_1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

x_train_lm_2=x_train_lm_1.drop(['Floor'],axis=1)
lr_2=sm.OLS(y_train,x_train_lm_2).fit()
lr_2.params
lr_2.summary()

x_test_lm_2=x_test_lm_1.drop(['Floor'],axis=1)
y_pred_lm_2=lr_2.predict(x_test_lm_2)
r2_test_2=r2_score(y_test,y_pred_lm_2)

vif = pd.DataFrame()
vif['Features'] = x_train_lm_2.columns
vif['VIF'] = [variance_inflation_factor(x_train_lm_2.values, i) for i in range(x_train_lm_2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

x_train_lm_3=x_train_lm_2.drop(['Corner'],axis=1)
lr_3=sm.OLS(y_train,x_train_lm_3).fit()
lr_3.params
lr_3.summary()

x_test_lm_3=x_test_lm_2.drop(['Corner'],axis=1)
y_pred_lm_3=lr_3.predict(x_test_lm_3)
r2_test_3=r2_score(y_test,y_pred_lm_3)

vif = pd.DataFrame()
vif['Features'] = x_train_lm_3.columns
vif['VIF'] = [variance_inflation_factor(x_train_lm_3.values, i) for i in range(x_train_lm_3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

