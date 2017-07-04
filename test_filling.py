from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd
import numpy as np
import random
df = pd.DataFrame([[32, 1, 3, 5, np.nan, 154],
                   [2, 1, np.nan, 5, 2, 154],
                   [2, 1, 3, 5, 13, 154],
                   [np.nan, 1, np.nan, np.nan, np.nan, np.nan],
                   [1, 4, np.nan, 5, 34, 169],
                   [random.randint(0, 100), random.randint(0, 100), 2, 5, random.randint(0, 100), 169],
                   [random.randint(0, 100), random.randint(0, 100), 2, 5, random.randint(0, 100), random.randint(0, 100)],
                   [random.randint(0, 100), random.randint(0, 100), 2, 5, random.randint(0, 100), 169],
                   [random.randint(0, 100), random.randint(0, 100), 2, 5, random.randint(0, 100), random.randint(0, 100)],
                   [random.randint(0, 100), random.randint(0, 100), 2, 5, random.randint(0, 100), 169],
                   [random.randint(0, 100), random.randint(0, 100), 2, 5, random.randint(0, 100), 169],
                   [random.randint(0, 100), random.randint(0, 100), 2, 5, random.randint(0, 100), random.randint(0, 100)],
                   [random.randint(0, 100), random.randint(0, 100), 2, 5, random.randint(0, 100), random.randint(0, 100)],
                   [random.randint(0, 100), random.randint(0, 100), 2, 5, random.randint(0, 100), 169],
                   [random.randint(0, 100), random.randint(0, 100), 2, 5, random.randint(0, 100), 169]])

print(df)

def imputer(dataframe):
    for i in dataframe.columns:
        print(i)
        tm = dataframe.copy()
        ls_tr_f = tm.dropna(axis=1).columns.values
        tm.dropna(axis=0, inplace=True)
        tm = tm[ls_tr_f]
        if len(dataframe[i].unique()) > 5:
            forest = ExtraTreesRegressor(n_estimators=100)
        else:
            forest = ExtraTreesClassifier(n_estimators=100)
        forest.fit(tm, dataframe.iloc[tm.index][i])
        for id, j in enumerate(dataframe[i]):
            if np.isnan(j):
                # print(i, j)
                dataframe.loc[id, i] = forest.predict(dataframe[ls_tr_f].iloc[id].reshape(1, -1))
    return dataframe
print(imputer(df.copy()))

df_train = pd.read_csv('ml5/train.csv', delimiter=';', na_values=None)
df_test = pd.read_csv('ml5/test.csv', delimiter=';', na_values=None)
df_test.fillna(df_test.mean(), inplace=True)
df_sub = pd.read_csv('ml5/submit.csv', delimiter=';', names=[0])

df_train = pd.concat([df_train, df_test], axis=0, ignore_index=True)
df_train = df_train.replace('None', np.nan)
df_train = df_train.replace(np.inf, np.nan)
df_train = df_train.astype('float')
df_train.drop(['id'], axis=1, inplace=True)
print(df_train)
rs_frame = imputer(df_train.drop(['cardio'], axis=1).copy())
rs_frame['cardio'] = df_train.cardio
df = rs_frame[rs_frame['cardio'].isnull()]
df.to_csv('test_fill.csv', index_label=True, index=True, header=True, sep=';')