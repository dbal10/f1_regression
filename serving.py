import pickle5 as pickle
import pandas as pd
import numpy as np
import data_preparation

LOGISTICREGRESSION = './models/logisticregression_f1.sav'
RANDOMFOREST = './models/randomforest_f1.sav'


class Serving:
    def __init__(self):

        self.drivers_df = pd.read_csv('./f1db_csv/drivers.csv')

        with open(LOGISTICREGRESSION, 'rb') as f:
            self.logisticRegr = pickle.load(f)


        with open(RANDOMFOREST, 'rb') as f:
            self.random_forest = pickle.load(f)


    def predict_driver_position_lr(self,row):
        '''
        Predicts result result of driver by logstic regression.
        '''

        return self.logisticRegr.predict_proba(row)


    def prdict_driver_position_rf(self, row):
        '''
        Predicts result result of driver by random forest classifier.
        '''
        return self.random_forest.predict_proba(row)

    def predict_race_lr(self, race_id):
        '''
        Predicts result result of driver by logstic regression.
        '''

        return self.predict_race_ranking(race_id, self.logisticRegr)


    def predict_race_rf(self, race_id):
        '''
        Predicts result result of driver by random forest classifier.
        '''
        return self.predict_race_ranking(race_id, self.random_forest)


    def predict_race_ranking(self, race_id, model):

        # wczytac dane z dataframe z date i raceid
        input_data_raw = pd.read_csv('./data_steps/train_df_all_cols_final.csv')
        input_data_date_df = input_data_raw[input_data_raw.raceId == race_id]

        info_actualrace = self.get_actual_race_info(race_id)

        # przetworzyc dane dla modelu
        serving_data_df = data_preparation.prepare_data_for_serve(input_data_date_df)
        serving_data_df_X = serving_data_df.drop(['position'], axis = 1)

        if serving_data_df.empty:
            return pd.DataFrame(columns=['real_position', 'probability', 'driverId', 'predicted_position'])
        # uruchomic clf.predict_proba
        results = pd.DataFrame(model.predict_proba(serving_data_df_X.values),columns=['probability', 'pos1', 'pos2', 'pos3'])

        # dociagnac id kierwocow
        results = results[['probability']]
        results['driverId'] = input_data_date_df.loc[:, ['driverId']].values


        results = results.merge(info_actualrace, left_on='driverId', right_on='driverId')


        # posortowac rezultaty
        results = results.sort_values('probability', ascending=False)
        results.insert(loc=0, column='predicted_position', value=np.arange(start = 1,stop = len(results) + 1))

        # zwrocic rezultaty w tabeli z parwdopodobeinstwami
        results = self.join_drivers(results)

        return results

    def get_actual_race_info(self, race_id):
        infodata_results_df = pd.read_csv("./f1db_csv/results.csv", delimiter=",")
        infodata_results_df = infodata_results_df[['raceId', 'driverId', 'position']]
        info_actualrace = infodata_results_df[infodata_results_df['raceId'] == race_id]
        info_actualrace.drop(['raceId'], inplace=True, axis=1)
        return info_actualrace

    def join_drivers(self, results):
        return  pd.merge(results, self.drivers_df[['driverId','forename', 'surname']], on = 'driverId')

def test_f1_serving(n):
    serving = Serving()
    return serving.predict_race_lr(n)



