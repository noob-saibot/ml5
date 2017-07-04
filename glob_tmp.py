from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd
import numpy as np
import random

df_test = pd.read_csv('ml5/test.csv', delimiter=';', na_values=None)
df_test = df_test.replace('None', np.nan)
df_test2 = pd.read_csv('test_fill.csv', delimiter=';', na_values=None)
df_test.drop(['id'], axis=1, inplace=True)
# df_test2.drop(['id'], axis=1, inplace=True)


for i in df_test.columns:
    for id, j in enumerate(df_test[i]):
        for jd in df_test.columns:
            print(jd, df_test[jd][j], df_test2[jd][j])