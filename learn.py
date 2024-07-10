from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, dcg_score, ndcg_score
from sklearn.metrics import confusion_matrix
import pickle5 as pickle
import pandas as pd
import numpy as np

def splitted(train_df, y_col =' position', test_size = 0.2):
    y = train_df[['position']]
    X = train_df.drop('position', axis='columns')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=49)
    return X_train.values, X_test.values, y_train.values, y_test.values

def time_series_splitted(train_df, y_col='position', test_fraction=0.2):

    dataset_size = train_df.shape[0]
    test_fraction = int(test_fraction * dataset_size)
    tscv = TimeSeriesSplit(n_splits=2, test_size=test_fraction, gap=1)

    y = train_df[[y_col]].values
    X = train_df.drop(y_col, axis='columns').values

    for train_index, test_index in tscv.split(X):

        print("TRAIN:", len(train_index), "TEST:", len(test_index))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test


def log_reg_eval(x,y):
    logisticRegr = None

    filename = './models/logisticregression_f1.sav'
    with open(filename, 'rb') as f:
        logisticRegr = pickle.load(f)

    y_pred = logisticRegr.predict(x)

    print("Score Logistic Regression")
    print(classification_report(y, y_pred))
    print('Conf matrix')
    print(confusion_matrix(y, y_pred, normalize = 'all'))

def log_reg_train(X, x, Y, y, dump = False):
    clf = LogisticRegression()
    clf.fit(X,Y)

    filename = './models/logisticregression_f1.sav'
    if dump:
        pickle.dump(clf, open(filename, 'wb'))

    y_pred = clf.predict(x)

    print("Score Logistic Regression")
    print(classification_report(y, y_pred))
    print('Conf matrix')
    print(confusion_matrix(y, y_pred, normalize = 'all'))
    return clf

def rf_eval(x,y):
    clf = None

    # save the model to disk
    filename = './models/randomforest_f1.sav'
    with open(filename, 'rb') as f:
        clf = pickle.load(f)

    y_pred = clf.predict(x)

    print("Score Random Forest")
    print(classification_report(y, y_pred))
    print('Conf matrix')
    print(confusion_matrix(y, y_pred, normalize = 'all'))

    return clf

def rf_train(X, x, Y, y, dump = False):

    class_value_counts = pd.DataFrame(Y).value_counts(sort=False)
    # odwrotne wagowanie klas
    weights = ((1/class_value_counts.values) / np.sum(1/class_value_counts.values ))

    clf = BalancedRandomForestClassifier( random_state=0, class_weight=dict(zip([1,2,3,4],list(weights))))
    clf.fit(X,Y)

    filename = './models/logisticregression_f1.sav'
    if dump:
        pickle.dump(clf, open(filename, 'wb'))

    y_pred = clf.predict(x)


    print("Score Random Forest Classifier")
    print(classification_report(y, y_pred))
    print('Conf matrix')
    print(confusion_matrix(y, y_pred, normalize = 'all'))

    return clf




