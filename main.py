import pandas as pd
import numpy as np
import data_preparation
import seaborn as sn
from matplotlib import pyplot as plt
import learn
import metrics


def train_models(p_train_df):
    train_df = p_train_df

    if train_df is None:
        train_df = pd.read_csv("./data_steps/train_df_final.csv")
        train_df_all_cols = pd.read_csv("./data_steps/train_df_all_cols_final.csv")

    #potrzebne do sprawdzenia które wyscigi sa w zbiorze testowym
    X_train_races, X_test_races, _, _ = learn.time_series_splitted(train_df_all_cols[['raceId', 'driverId', 'position']])
    pd.DataFrame(np.unique(X_train_races[:, 0]).transpose()).to_csv('./data_steps/train_races.csv', index = False)
    pd.DataFrame(np.unique(X_test_races[:, 0]).transpose()).to_csv('./data_steps/test_races.csv', index = False)


    X_train, X_test, y_train, y_test = learn.time_series_splitted(train_df)

    rf = learn.rf_train(X_train, X_test, y_train, y_test, dump=True)
    logreg = learn.log_reg_train(X_train, X_test, y_train, y_test, dump=True)
    print("Score rankings Logistic Regression")
    metrics.compute_dcgs(X_test, X_test_races, logreg)
    print("Score rankings Random Forest Classifier")
    metrics.compute_dcgs(X_test, X_test_races, rf)


def create_train_data(save_results = False):
    train_df,train_df_all_cols = data_preparation.prepare_for_train()
    if save_results:
        train_df.to_csv('./data_steps/train_df_final.csv', index=False)
        train_df_all_cols.to_csv('./data_steps/train_df_all_cols_final.csv', index=False)


    return train_df



def show_cor_matrix(train_df):
    corrMatrix = train_df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()


def load_options():
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.options.mode.chained_assignment = None
    pass


if __name__ == '__main__':
    load_options()
    #train_df = create_train_data(save_results= True)
    train_models(None)


