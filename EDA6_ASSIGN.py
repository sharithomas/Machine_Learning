# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:41:51 2020

@author: SHARI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv(r"C:\Users\SHARI\Desktop\PYTHON PGMS\lectures\python_set2\machine_learning\EDA\assignment_works\marks_6_students_1.csv")
df.columns
df.dtypes