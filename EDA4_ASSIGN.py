# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:41:51 2020

@author: SHARI
"""
 #Housing Price,Nan by impute,outlier by capping, Label Encoding, sklearn for train_test data,VIF
import pandas as pd
import numpy as np
import seaborn as sns #visulaization
import matplotlib.pyplot as plt  #visulaization

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,r2_score 

sns.set(color_codes=True)
df=pd.read_csv(r"C:\Users\SHARI\Desktop\PYTHON PGMS\lectures\python_set2\machine_learning\EDA\Housing_Price_170720_test.csv")
df.columns
df.dtypes

df.isnull().mean()*100
df.shape

df.dtypes
num_var_list=['Length','Width','Floor','Balcony_Area']
cat_var_list=['Corner','Special_Feature']

df.Special_Feature.mode()
df.Special_Feature.mode().iloc[0]

#handling Nan by impute(mean,mode,median)
def impute_na_num(df,variable):
    return df[variable].fillna(df[variable].mean())

for var in num_var_list:
    print(var)
    df[var]=impute_na_num(df,var)
    
df.isnull().mean()

def impute_na_cat(df,variable):
    return df[variable].fillna(df[variable].mode().iloc[0]) # to access value Pool_Facing from series  0    Pool_Facing

for var in cat_var_list:
    print(var)
    df[var]=impute_na_cat(df,var)
    
df.isnull().mean()

sns.boxplot(x=df['Width'])
sns.boxplot(x=df['Length'])
sns.boxplot(x=df['Floor'])

df_out=df[['Length','Width']]

#handling outliers by capping
lb=df_out.quantile(0.1)
ub=df_out.quantile(0.9)

df_out=df_out.clip(lower=df_out.quantile(0.1),upper=df_out.quantile(0.9),axis=1) # capping theoutliers values f length and width
df=df.drop(['Length','Width'],axis=1) # drop length and width from dataframe df
df=pd.concat([df,df_out],axis=1,join='inner') # concatenate df and df_out, join:inner(intersection) & outer(union)

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

x_train_lm_1=x_train_lm.drop(['Floor'],axis=1)
lr_1=sm.OLS(y_train,x_train_lm_1).fit()
lr_1.params
lr_1.summary()

x_test_lm_1=x_test_lm.drop(['Floor'],axis=1)
y_pred_lm_1=lr_1.predict(x_test_lm_1)
r2_test_1=r2_score(y_test,y_pred_lm_1)

vif = pd.DataFrame()
vif['Features'] = x_train_lm_1.columns
vif['VIF'] = [variance_inflation_factor(x_train_lm_1.values, i) for i in range(x_train_lm_1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

x_train_lm_2=x_train_lm_1.drop(['Corner'],axis=1)
lr_2=sm.OLS(y_train,x_train_lm_2).fit()
lr_2.params
lr_2.summary()

x_test_lm_2=x_test_lm_1.drop(['Corner'],axis=1)
y_pred_lm_2=lr_2.predict(x_test_lm_2)
r2_test_2=r2_score(y_test,y_pred_lm_2)

vif = pd.DataFrame()
vif['Features'] = x_train_lm_1.columns
vif['VIF'] = [variance_inflation_factor(x_train_lm_1.values, i) for i in range(x_train_lm_1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


x_train_lm_3 = x_train_lm_2.drop(["Special_Feature"], axis=1)
lr_3 = sm.OLS(y_train, x_train_lm_3).fit()
lr_3.params
print(lr_3.summary())

x_test_lm_3 = x_test_lm_2.drop(["Special_Feature"], axis=1)
y_pred_test_lm_3 = lr_3.predict(x_test_lm_3)
r2_test_3 = r2_score(y_test, y_pred_test_lm_3)


