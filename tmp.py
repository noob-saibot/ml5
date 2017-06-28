import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, cross_val_predict
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, ExtraTreesRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, log_loss
from sklearn.model_selection import ShuffleSplit
import pickle
import numpy as np
import matplotlib.pyplot as plt
from seaborn import set_style
set_style()

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

def main():
    df_train = pd.read_csv('ml5/train.csv', delimiter=';')
    df_test = pd.read_csv('ml5/test.csv', delimiter=';')

    df_train = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    df_train = df_train.replace('None', np.nan)
    for i in df_train.drop(['id', 'cardio'], axis=1).dropna().columns:
        df_train[i] = df_train[i].astype('float')
        cond = (df_train[i] > df_train[i].quantile(0.999)) | (df_train[i] < df_train[i].quantile(0.001))
        print(df_train[i][cond])
        df_train[i][cond] = df_train[i].mean()
        #print(df_train[i].dropna())
        # print(i)
        # df_train[i].astype('float').hist()
        # plt.show()


if __name__ == '__main__':
    main()
