# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:22:41 2020

@author: windows
"""

#DataSet:marks_6_students_1,
#Drop NaN, remove outlier, OneHat encoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("marks_6_students_1.csv")
df.columns
df.dtypes

df.isnull().mean()*100
num_list=['Engg Per','Uni_Grade','Experinece','Interview_Score','Test_Marks','No_Patents']
cat_list=['Qualification']
df.shape

#Handling Nan by removing Nan
df=df.dropna()
df.shape

#handling outliers
sns.boxplot(x=df['Engg Per'])
sns.boxplot(x=df['Uni_Grade'])
sns.boxplot(x=df['Experinece'])
sns.boxplot(x=df['Interview_Score'])
sns.boxplot(x=df['Test_Marks'])
sns.boxplot(x=df['No_Patents'])

Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1
print(IQR)

df=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR))).any(axis=1)]
df.shape

sns.boxplot(x=df['Engg Per'])
sns.boxplot(x=df['Uni_Grade'])
sns.boxplot(x=df['Experinece'])
sns.boxplot(x=df['Interview_Score'])
sns.boxplot(x=df['Test_Marks'])
sns.boxplot(x=df['No_Patents'])
df.shape

#HANDLING CATEGORICAL VARIABLES- ONEHAT ENCODING   
dummy=pd.get_dummies(df['Qualification'])
df=pd.concat([df,dummy],axis=1)
df=df.drop(['Qualification'],axis=1)
df.shape

#train test data processing
from sklearn.model_selection import train_test_split

df_train,df_test=train_test_split(df,train_size=0.7,test_size=0.3,random_state=0)
y_train=df_train.pop('Salary')
x_train=df_train

#fit train data in model and predict target value
import statsmodels.api as sm
x_train_lm=sm.add_constant(x_train)

lr=sm.OLS(y_train,x_train_lm).fit()
lr.params
lr.summary()

y_test=df_test.pop('Salary')
x_test=df_test

x_test_lm=sm.add_constant(x_test)
x_test_lm.shape

y_pred_lm=lr.predict(x_test_lm)
r2_test=r2_score(y_test,y_pred_lm)

#VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = x_train_lm.columns
vif['VIF'] = [variance_inflation_factor(x_train_lm.values, i) for i in range(x_train_lm.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

x_train_lm_1=x_train_lm.drop('Interview_Score',axis=1)
lr_1=sm.OLS(y_train,x_train_lm_1).fit()
lr_1.summary()

x_test_lm_1=x_test_lm.drop('Interview_Score',axis=1)
y_pred_lm_1=lr_1.predict(x_test_lm_1)
r2_test_1=r2_score(y_test,y_pred_lm_1)

x_train_lm_2=x_train_lm_1.drop('Test_Marks',axis=1)
lr_2=sm.OLS(y_train,x_train_lm_2).fit()
lr_2.summary()

x_test_lm_2=x_test_lm_1.drop('Test_Marks',axis=1)
y_test_lm_2=lr_2.predict(x_test_lm_2)
r2_test_2=r2_score(y_test,y_test_lm_2)




