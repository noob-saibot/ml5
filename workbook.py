import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, cross_val_predict
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, ExtraTreesRegressor, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import ShuffleSplit
import pickle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import xgbfir


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

    df_train = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    df_train['opt_w'] = (50 + 0.75 * (df_train['height'] - 150) + (df_train['age']/365 - 20)) / df_train['weight']

    df_train.drop(['weight'], axis=1, inplace=True)

    # df_train['opt_ap'] = df_train['ap_hi']/df_train['ap_lo']

    df_train.drop(['id'], axis=1, inplace=True)

    df_train['gender'] = df_train['gender'].astype('str')

    df_train = pd.get_dummies(df_train, columns=['gender'])

    df_train = df_train.convert_objects(convert_numeric=True)

    df_train = df_train.replace(np.inf, np.nan)

    df_train.fillna(df_train.drop(['cardio'], axis=1).mean(), inplace=True)

    df_train = df_train.replace('None', np.nan)
    for i in df_train.drop(['cardio'], axis=1).dropna().columns:
        df_train[i] = df_train[i].astype('float')
        cond = (df_train[i] > df_train[i].quantile(0.99)) | (df_train[i] < df_train[i].quantile(0.01))
        df_train[i][cond] = df_train[i].mean()

    # df_train['age'] = df_train['age']/365

    # for i in df_train.drop(['cardio'], axis=1).columns:
    #     df_train[i] = (df_train[i] - df_train[i].mean()) / (df_train[i].max() - df_train[i].min())

    print(df_train)

    # df_train['opt_ap_hi'] = (109 + (0.5 * df_train['age'] / 365) + (0.1 * df_train['weight'])) / df_train['ap_hi']
    #
    # df_train['opt_ap_lo'] = (63 + (0.1 * df_train['age'] / 365) + (0.15 * df_train['weight'])) / df_train['ap_lo']

    df_test = df_train[df_train['cardio'].isnull()]

    df_train.dropna(axis=0, inplace=True)

    dx, dxt, dy, dyt = train_test_split(df_train.drop(['cardio'], axis=1), df_train.cardio, train_size=0.6)

    # model = ExtraTreesClassifier(n_estimators=1000, n_jobs=3, max_features='sqrt', max_depth=10)
    # model = AdaBoostClassifier()
    # model = LogisticRegression()
    # model = SVC(probability=True)
    # model = GradientBoostingClassifier()
    # model = XGBClassifier(objective='binary:logistic')

    print(dx.shape, dxt.shape)
    # title = "Learning Curves"
    # cv = ShuffleSplit(n_splits=6, test_size=0.2)
    # plot_learning_curve(model, title, dx, dy, cv=cv, n_jobs=3)

    # plt.show()

    # print(-cross_val_score(model, df_train.drop(['cardio'], axis=1), df_train.cardio, cv=6, scoring='neg_log_loss', n_jobs=3).mean())

    model.fit(dx, dy)

    print(list(zip(dx.columns,model.feature_importances_)))

    pred = model.predict_proba(dx)
    print('#################PRED:', log_loss(dy, pred[:,1]))
    print('#################ROC: ', roc_auc_score(dy, pred[:,1]))
    counter(pd.DataFrame(pred), 0)

    pred_t = model.predict_proba(dxt)
    print('#################PRED_T: ', log_loss(dyt, pred_t[:,1]))
    print('#################ROC: ', roc_auc_score(dyt, pred_t[:,1]))
    counter(pd.DataFrame(pred_t), 0)

    df_sub[0] = model.predict_proba(df_test.drop(['cardio'], axis=1))
    print('#################SUB: ')
    counter(df_sub, 0)

    # print(model.predict_proba(df_test.drop(['cardio'], axis=1)))
    # print(df_sub)

    df_sub.to_csv('sub_base.csv', index_label=False, index=False, header=False, sep=';')

if __name__ == '__main__':
    main()
