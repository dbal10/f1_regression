import pandas as pd
import numpy as np
from sklearn.metrics import dcg_score, ndcg_score


def predict_race_ranking(data_race):

    # posortowac rezultaty, ascending by probabilty win & descending rest by probability loose
    results = data_race.sort_values(['posistions-first-proba','posistions-last-proba'], ascending=[False,True])

    results.insert(loc=0, column='predicted_position', value=np.arange(start = 1,stop = len(results) + 1))

    return results


def compute_dcgs(X_test, X_test_races, model=None):
    """
    X_test  - tablica ze zbiorem testowym dla modelu (nie zawiera raceid i driver id)
    X_test_races - tablica zawierajaca race id i driver odpowiadajace w kolejności tym z X_test
    model - model predykcyjny
    """
    races_df = pd.read_csv("./f1db_csv/races.csv", delimiter=",")
    results_df = pd.read_csv("./f1db_csv/results.csv", delimiter=",")
    results_df = results_df[['raceId', 'driverId', 'position']]


    df_real_dat = pd.DataFrame(X_test_races, columns=['raceId', 'driverId'])

    # dane rzeczywiste służące do porównania wyników
    df_real_dat = pd.merge(df_real_dat, results_df, how='left', left_on=['raceId', 'driverId'], right_on=['raceId', 'driverId'])

    # przewidywanie prawdopodobieństw do rankingu
    df_predicions =pd.DataFrame( model.predict_proba(X_test),columns=['posistions-first-proba', 'pos1', 'pos2', 'posistions-last-proba'])

    df_to_dacgs  = pd.concat([df_real_dat,df_predicions], ignore_index=True, axis = 1)

    df_to_dacgs.set_axis(["raceId", 'driverId',"real_position",'posistions-first-proba', 'pos1', 'pos2', 'posistions-last-proba'], axis=1, inplace=True)
    # zamiena wszystkich kolumn na numeric
    df_to_dacgs["real_position"] = pd.to_numeric(df_to_dacgs["real_position"])

    group_by_raceid = df_to_dacgs.groupby("raceId")


    # lista metryk dcg do obliczenia średniej
    dcgs = []
    # lista metryk ndcg do obliczenia średniej
    n_dcgs = []

    for race_id, data_race in group_by_raceid:
        results = predict_race_ranking(data_race=data_race)

        if results.empty:
            continue
        only_drivers_finished = results[results['real_position'] > 1]
        real_positions = only_drivers_finished['real_position'].values.astype(int)
        prodicted_positions = only_drivers_finished['predicted_position'].values
        if prodicted_positions.shape[0] > 1:
            dcg = dcg_score([real_positions], [prodicted_positions])
            n_dgc = ndcg_score([real_positions], [prodicted_positions])
            n_dcgs.append(n_dgc)
            dcgs.append(dcg)

            print("NDGC for race: {} is {}".format(races_df[races_df.raceId == race_id], n_dgc))
            print("{}".format(results))

    dcgs_ndarr = np.array(dcgs)
    ndcgs_ndarr = np.array(n_dcgs)
    print('MEAN DCG for all races: ', dcgs_ndarr.mean())
    print('with standard deviation : ', dcgs_ndarr.std())
    print('===========')
    print('MEAN Normalized DCG for all races: ', ndcgs_ndarr.mean())
    print('with standard deviation : ', ndcgs_ndarr.std())

    return None