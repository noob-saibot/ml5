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
from xgboost import XGBClassifier


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
    df_train = pd.read_csv('ml5/train.csv', delimiter=';', na_values=None)
    df_test = pd.read_csv('ml5/test.csv', delimiter=';', na_values=None)
    df_sub = pd.read_csv('ml5/submit.csv', delimiter=';', names=[0])

    df_train = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    df_train.drop(['id'], axis=1, inplace=True)

    df_train['gender'] = df_train['gender'].astype('str')

    df_train = pd.get_dummies(df_train, columns=['gender'])

    df_train = df_train.convert_objects(convert_numeric=True)

    df_train.fillna(df_train.drop(['cardio'], axis=1).mean(), inplace=True)

    df_train = df_train.replace('None', np.nan)
    for i in df_train.drop(['cardio'], axis=1).dropna().columns:
        df_train[i] = df_train[i].astype('float')
        cond = (df_train[i] > df_train[i].quantile(0.9999)) | (df_train[i] < df_train[i].quantile(0.0001))
        df_train[i][cond] = df_train[i].mean()

    df_train['age'] = df_train['age']/365

    for i in df_train.drop(['cardio'], axis=1).columns:
        df_train[i] = (df_train[i] - df_train[i].mean()) / (df_train[i].max() - df_train[i].min())

    print(df_train)

    df_test = df_train[df_train['cardio'].isnull()]

    df_train.dropna(axis=0, inplace=True)

    dx, dxt, dy, dyt = train_test_split(df_train.drop(['cardio'], axis=1),
                                        df_train.cardio,
                                        test_size=.2)

    model = ExtraTreesClassifier(max_depth=12)
    model = GradientBoostingClassifier()

    print(dx.shape, dxt.shape)
    title = "Learning Curves"
    cv = ShuffleSplit(n_splits=6, test_size=0.2)
    plot_learning_curve(model, title, dx, dy, cv=cv, n_jobs=3)

    plt.show()

    print(-cross_val_score(model, df_train.drop(['cardio'], axis=1), df_train.cardio, cv=6, scoring='neg_log_loss', n_jobs=3).mean())

    model.fit(dx, dy)

    pred = model.predict_proba(dx)

    pred_t = model.predict_proba(dxt)

    print(log_loss(dy, pred))
    print(log_loss(dyt, pred_t))
    df_sub[0] = model.predict_proba(df_test.drop(['cardio'], axis=1))
    # print(model.predict_proba(df_test.drop(['cardio'], axis=1)))
    # print(df_sub)

    df_sub.to_csv('sub_base.csv', index_label=False, index=False, header=False, sep=';')

if __name__ == '__main__':
    main()