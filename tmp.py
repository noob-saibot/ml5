import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, cross_val_predict
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, ExtraTreesRegressor, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import ShuffleSplit
import pickle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import xgbfir
import seaborn
seaborn.set_style()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_log_loss')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def counter(df, dst):
    for i in range(90, 100, 1):
        print(i/100.0, df[dst][np.round(df[dst], 2) == i/100.0].size, '###', (100-i)/100.0, df[dst][np.round(df[dst], 2) == (100-i)/100.0].size)

def main():
    df_train = pd.read_csv('ml5/train.csv', delimiter=';', na_values=None)
    df_test = pd.read_csv('ml5/test.csv', delimiter=';', na_values=None)
    df_test.fillna(df_test.mean(), inplace=True)
    df_sub = pd.read_csv('ml5/submit.csv', delimiter=';', names=[0])

    df_train['opt_w'] = (50 + 0.75 * (df_train['height'] - 150) + (df_train['age'] / 365 - 20)) / df_train['weight']

    df_train['imt'] = df_train['weight'] / (df_train['height'] ** 2)

    df_train['opt_ap_hi'] = (109 + (0.5 * df_train['age'] / 365) + (0.1 * df_train['weight'])) / df_train['ap_hi']

    df_train['opt_ap_lo'] = (63 + (0.1 * df_train['age'] / 365) + (0.15 * df_train['weight'])) / df_train['ap_lo']

    df_train['minus'] = df_train['ap_hi'] - df_train['ap_lo']

    print(df_train['ap_lo'].min(), df_train['ap_lo'].max())
    print(df_train['ap_hi'].min(), df_train['ap_hi'].max())

    df_train['ap_lo'] = np.abs(df_train['ap_lo'])
    df_train['ap_hi'] = np.abs(df_train['ap_hi'])

    print(df_train['ap_lo'].min(), df_train['ap_lo'].max())
    print(df_train['ap_hi'].min(), df_train['ap_hi'].max())

    df_train.drop(['id'], axis=1, inplace=True)

    df_train = df_train.replace('None', np.nan)
    for i in df_train.drop(['cardio'], axis=1).dropna().columns:
        df_train[i] = df_train[i].astype('float')
        cond = (df_train[i] > df_train[i].quantile(0.95)) | (df_train[i] < df_train[i].quantile(0.05))
        tm = df_train.groupby(i)['age'].nunique()
        if tm.size == 2 or tm.size == 3:
            df_train.loc[cond, i] = tm.idxmax()
        else:
            df_train.loc[cond, i] = df_train[i].mean()

    print(df_train['ap_lo'].min(), df_train['ap_lo'].max())
    print(df_train['ap_hi'].min(), df_train['ap_hi'].max())

    df_train = df_train.replace(np.inf, np.nan)

    df_train.fillna(df_train.mean(), inplace=True)

    df_train = df_train.astype('float')

    i = 'opt_w'
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

if __name__ == '__main__':
    main()
