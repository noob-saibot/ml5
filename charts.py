from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd
import numpy as np
import random
import seaborn
import matplotlib.pyplot as plt
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

def imputer_out(dataframe):
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
                print(id)
                # print(dataframe.loc[id, i], forest.predict(dataframe[ls_tr_f].iloc[id].reshape(1, -1)))
                dataframe.loc[id, i] = forest.predict(dataframe[ls_tr_f].iloc[id].reshape(1, -1))
    return dataframe

# df_train = pd.read_csv('train_out.csv', delimiter=';', na_values=None)
df_train = pd.read_csv('ml5/train.csv', delimiter=';', na_values=None)

# for i in df_train.drop(['cardio'], axis=1).dropna().columns:
#         df_train[i] = df_train[i].astype('float')
#         cond = (df_train[i] > df_train[i].quantile(0.999)) | (df_train[i] < df_train[i].quantile(0.001))
#         df_train[i][cond] = np.nan
# rs = imputer_out(df_train.drop(['cardio'], axis=1))
# rs['cardio'] = df_train.cardio
# rs.to_csv('train_out.csv', index_label=True, index=True, header=True, sep=';')
i = 'age'
for j in df_train.columns:
    if j != i and j != 'cardio':
        print(i, j)
        print(df_train[[i, j]].head(10))
        seaborn.pairplot(df_train[[i, j, 'cardio']],
                         size=3, hue="cardio", palette="husl", diag_kind="kde",
                         vars=[i, j])
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()