from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
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

for i in df.columns:
    tm = df.copy()
    ls_tr_f = tm.dropna(axis=1).columns.values
    tm.dropna(axis=0, inplace=True)
    tm = tm[ls_tr_f]
    if len(df[i].unique()) > 5:
        forest = KNeighborsRegressor(n_neighbors=3)
    else:
        forest = KNeighborsClassifier(n_neighbors=3)
    forest.fit(tm, df.iloc[tm.index][i])
    for id, j in enumerate(df[i]):
        if np.isnan(j):
            df.loc[id, i] = forest.predict(df[ls_tr_f].iloc[id].reshape(1, -1))
print(df)