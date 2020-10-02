# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:15:25 2020

@author: shrav
"""

#1. Importing the required libraries for EDA

import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
  
sns.set(color_codes=True)

#2. Loading the data into the data frame

df = pd.read_csv("Housing_Price_170720_test.csv")

df.columns

#df.head(5)

df.dtypes

df.isnull().mean()*100

df.shape

df=df.dropna()

df.shape

sns.boxplot(x=df['Length'])

sns.boxplot(x=df['Width'])

sns.boxplot(x=df['Floor'])

list_out=["Length", "Width"]

def remove_outlier(df, col_name):
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
    return df_out

for var in list_out:
    df=remove_outlier(df,var)
    
df.shape
    

df['Area']=df['Length']*df['Width']

df.head()

df=df.drop(['Length','Width'],axis=1)

df

df.shape

from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 

df['Special_Feature']= le.fit_transform(df['Special_Feature']) 

df.head()

df.columns

df=df.drop(['Id'],axis=1)

df.shape



from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

num_vars = ['Floor',  'Special_Feature', 'Balcony_Area', 'Price', 'Area']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

type(df_train)

df_test[num_vars] = scaler.transform(df_test[num_vars])


#
#
#
### Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
##num_vars = ['Floor','Balcony_Area', 'Area']
##
##df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
##
##df_test[num_vars] = scaler.transform(df_test[num_vars])


y_train = df_train.pop('Price')
X_train = df_train

import statsmodels.api as sm

# Add a constant
X_train_lm = sm.add_constant(X_train)

# Create a first fitted model
lr = sm.OLS(y_train, X_train_lm).fit()

lr.params

print(lr.summary())


y_test = df_test.pop('Price')
X_test = df_test

X_test_lm = sm.add_constant(X_test)

y_pred_lm = lr.predict(X_test_lm)

r2_test = r2_score(y_test, y_pred_lm)

plt.figure(figsize=(10,5))
c= X_train_lm.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X_train_lm.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm.values, i) for i in range(X_train_lm.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif



X_train_lm_1 = X_train_lm.drop(["Balcony_Area"], axis=1)

lr_1 = sm.OLS(y_train, X_train_lm_1).fit()

lr_1.params

print(lr_1.summary())

X_test_lm_1 = X_test_lm.drop(["Balcony_Area"], axis=1)
y_pred_test_lm_1 = lr_1.predict(X_test_lm_1)

r2_test_1 = r2_score(y_test, y_pred_test_lm_1)

vif = pd.DataFrame()
vif['Features'] = X_train_lm_1.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm_1.values, i) for i in range(X_train_lm_1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

X_train_lm_2 = X_train_lm_1.drop(["Floor"], axis=1)

lr_2 = sm.OLS(y_train, X_train_lm_2).fit()

lr_2.params

print(lr_2.summary())

X_test_lm_2 = X_test_lm_1.drop(["Floor"], axis=1)
y_pred_test_lm_2 = lr_2.predict(X_test_lm_2)

r2_test_2 = r2_score(y_test, y_pred_test_lm_2)

y_pred_train_lm_2 = lr_2.predict(X_train_lm_2)


fig = plt.figure()
sns.distplot((y_train - y_pred_train_lm_2))
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)


X_train_lm_3 = X_train_lm_2.drop(["Corner"], axis=1)

lr_3 = sm.OLS(y_train, X_train_lm_3).fit()

lr_3.params

print(lr_3.summary())

X_test_lm_3 = X_test_lm_2.drop(["Corner"], axis=1)
y_pred_test_lm_3 = lr_3.predict(X_test_lm_3)
type(y_pred_test_lm_3)

r2_test_3 = r2_score(y_test, y_pred_test_lm_3)

y_pred_train_lm_3 = lr_3.predict(X_train_lm_3)

#error are normally distributed
fig = plt.figure()
sns.distplot((y_train - y_pred_train_lm_3))
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)

#heteroscedasticity
fig = plt.figure()
sns.scatterplot(y_train,(y_train - y_pred_train_lm_3))
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)

#X_train_lm_4 = X_train_lm_3.drop(["const"], axis=1)
#
#lr_4 = sm.OLS(y_train, X_train_lm_4).fit()
#
#lr_4.params
#
#print(lr_4.summary())
#
#X_test_lm_4 = X_test_lm_3.drop(["const"], axis=1)
#y_pred_test_lm_4 = lr_4.predict(X_test_lm_4)
#
#r2_test_4 = r2_score(y_test, y_pred_test_lm_4)


####sklearn########

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print('Slope:' ,regressor.coef_)
print('Intercept:', regressor.intercept_)

# Predicting the Test set results
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)

r2_test = r2_score(y_test, y_pred_test)




X_train_1=X_train.drop(["Balcony_Area"],axis=1)

regressor_1 = LinearRegression()
regressor_1.fit(X_train_1, y_train)

print('Slope:' ,regressor_1.coef_)
print('Intercept:', regressor_1.intercept_)

X_test_1=X_test.drop(["Balcony_Area"],axis=1)

# Predicting the Test set results
y_pred_train_1 = regressor_1.predict(X_train_1)
y_pred_test_1 = regressor_1.predict(X_test_1)

r2_train_1 = r2_score(y_train, y_pred_train_1)

r2_test_1 = r2_score(y_test, y_pred_test_1)



X_train_2=X_train_1.drop(["Floor"],axis=1)

regressor_2 = LinearRegression()
regressor_2.fit(X_train_2, y_train)

print('Slope:' ,regressor_2.coef_)
print('Intercept:', regressor_2.intercept_)

X_test_2=X_test_1.drop(["Floor"],axis=1)

# Predicting the Test set results
y_pred_train_2 = regressor_2.predict(X_train_2)
y_pred_test_2 = regressor_2.predict(X_test_2)

r2_train_2 = r2_score(y_train, y_pred_train_2)

r2_test_2 = r2_score(y_test, y_pred_test_2)



X_train_3=X_train_2.drop(["Corner"],axis=1)

regressor_3 = LinearRegression()
regressor_3.fit(X_train_3, y_train)

print('Slope:' ,regressor_3.coef_)
print('Intercept:', regressor_3.intercept_)

X_test_3=X_test_2.drop(["Corner"],axis=1)

# Predicting the Test set results
y_pred_train_3 = regressor_3.predict(X_train_3)
y_pred_test_3 = regressor_3.predict(X_test_3)

r2_train_3 = r2_score(y_train, y_pred_train_3)

r2_test_3 = r2_score(y_test, y_pred_test_3)


#scalar.inverse_transform(dataframe)




























