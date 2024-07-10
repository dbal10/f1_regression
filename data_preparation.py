import datetime

import pandas as pd
import numpy as np
from datetime import date

from tqdm import tqdm

YEAR_FROM_CFG = 1990

TAKE_ONLY_FIRST_100 = False
TAKE_ONLY_FIRTS_ROWS = 100


def count_driver_fails(results_df):
    table = results_df[['driverId', 'position']]
    table = table.replace(r'\\N', 0, regex=True)
    table = table[table['position'] == 0]
    table = pd.pivot_table(table, index=['driverId'], values='position', aggfunc='count')
    table.rename({'position': 'driver_all_fails_count'}, axis=1, inplace=True)

    return table

def load_input_data():
    df = pd.read_csv('./data_steps/train_df_full.csv')

    df.drop(list(df.filter(regex='Unnamed.*')), axis=1, inplace=True)

    df.drop(['ls_driver_points', 'ls_constructor_points'],axis=1, inplace=True)

    df.drop_duplicates()

    return df

def clean_data(train_df):
    train_df = train_df[['grid','last_circuit_position', 'last_position', 'last_position2', 'last_position3', 'last_position4', 'last_position5','ls_constructor_points','ls_driver_points','position']]

    return train_df

def map_target(train_df, training  = True):

    train_df = train_df[pd.to_numeric(train_df['position'], errors='coerce').notnull()]
    if training:
        train_df.drop(train_df[train_df['position'] < 1].index, inplace = True)
    train_df.dropna()

    train_df['position'] = pd.to_numeric(train_df['position'])
    criteria = [train_df['position'].between(1,1), train_df['position'].between(2, 3), train_df['position'].between(4, 10), train_df['position'].between(10, 30)]
    values = [1,2,3,4]

    train_df['position'] = np.select(criteria, values, 0)
    print('Number of results by final postion ranges', train_df.groupby(['position'])['position'].count())
    return train_df

def prepare_for_train():

    results = get_results_df()
    results = results[['raceId', 'driverId', 'grid', 'position']]

    races_df = get_races_df()

    train_df = races_df.merge(results, how="outer", left_on='raceId', right_on='raceId',
                              suffixes=('_left', '_right'))

    print('Number of training records (races, results): ', train_df.shape)
    print('Number of results by postion',train_df.groupby(['position'])['position'].count())


    train_df['position'] = train_df['position'].replace(['\\N'], '-1')
    train_df['position'] = pd.to_numeric(train_df['position'])

    #sortowanie wynikow po datetime zapewnia poprawne wykonanie operacji last
    train_df["datetime"] = pd.to_datetime(train_df["date"])
    train_df = train_df.sort_values(by="datetime")

    train_df = append_driver_constructor_points(train_df)

    train_df = append_last_circuit_position(train_df)
    print('Number of training records last circuit position: ', train_df.shape)

    train_df = append_last_positions(train_df)

    all_cols_df = train_df.copy(deep=True)

    print('Number of training records last positions: ', train_df.shape)

    train_df = clean_data(train_df)

    print('Number of training records clean data: ', train_df.shape)
    train_df = map_target(train_df)
    print('Number of training records map target: ', train_df.shape)

    all_cols_df = map_target(all_cols_df)

    return train_df,all_cols_df


def prepare_data_for_serve(df):
    serving_df = clean_data(df)
    serving_df = map_target(serving_df, training=False)


    return serving_df

def append_last_positions(train_df, take_last = 5):
    '''

    Pobiera czas danego drivera dla dlanego race_id jaki uzyskał dla tego circuit_id w ostatnim wyscigu.

    petla for po race: wybiermay list driverow i dolaczamy circuit i date
    wybieramy z caego datframe po driver_id, cuircuit_id czasy wraz z datą, wyrzycic aktualny race id dla daty akt.

    => cuiruit_id, date, time wybieramy po  max(date) time
    appand do dataframe temp race_id, last_cuircuit_time

    :param train_df:
    :return:
    '''

    # print(train_df)

    temp_df = pd.DataFrame()


    for i, row in tqdm(train_df.iterrows()):
        on_driver_id = (train_df['driverId'] == row['driverId'])
        on_date = (train_df['datetime'] < row['datetime'])
        on_position_non_empty = (train_df['position'] >= 0)
        rr = train_df.loc[on_driver_id & on_position_non_empty & on_date ]
        r = {}
        if rr.shape[0] < take_last:
            continue
        try:
            if TAKE_ONLY_FIRST_100 and row['raceId'] > TAKE_ONLY_FIRTS_ROWS:
                continue
            #take max
            column_name = 'datetime'

            last_positions = []
            rows_to_exclude = []
            rows_to_exclude.append(rr)
            for pos_idx in range(1,take_last + 1):
                last_position = take_max(column_name, rows_to_exclude[pos_idx - 1])['position'].values[0]
                last_positions.append(last_position)
                rows_to_exclude.append(exclude_max(column_name, rows_to_exclude[pos_idx - 1]))
                if pos_idx == 1:
                    r['last_position'] = last_position
                else:
                    r['last_position' + str(pos_idx)] = last_position

                r['raceId'] = row['raceId']
                r['driverId'] = row['driverId']

            temp_df = temp_df.append(r, ignore_index=True)

        except Exception:
            pass

    if not temp_df.empty:
        new_df = pd.merge(train_df, temp_df, how='inner', left_on=[ 'raceId','driverId'], right_on=[ 'raceId','driverId'])
        return new_df

    else:
        return temp_df


def append_last_circuit_position(train_df):
    '''
    Pobiera czas danego drivera dla dlanego race_id jaki uzyskał dla tego circuit_id w ostatnim wyscigu.

    petla for po race: wybiermay list driverow i dolaczamy circuit i date
    wybieramy z caego datframe po driver_id, cuircuit_id czasy wraz z datą, wyrzycic aktualny race id dla daty akt.

    => cuiruit_id, date, time wybieramy po  max(date) time
    appand do dataframe temp race_id, last_cuircuit_time

    :param train_df:
    :return:
    '''

    # print(train_df)

    temp_df = pd.DataFrame()


    for i, row in tqdm(train_df.iterrows()):
        try:
            # TODO: debug
            if TAKE_ONLY_FIRST_100 and row['raceId'] > TAKE_ONLY_FIRTS_ROWS:
                continue

            on_cuicuit_id = (train_df['circuitId'] == row['circuitId'])
            on_driver_id = (train_df['driverId'] == row['driverId'])
            on_datetime = (train_df['datetime'] < row['datetime'])
            on_position_non_empty = (train_df['position'] >= 0)
            rr = train_df.loc[on_driver_id & on_cuicuit_id & on_position_non_empty & on_datetime ]
            r = {}

            last_circuit_num = 1
            if rr.shape[0] < last_circuit_num:
                continue

            #take max
            column_name = 'datetime'
            max1 = take_max(column_name, rr)
            last_position_1 = max1['position'].values[0]

            # TODO Zmieniono na 1 ostatni pozycj dla zwiekszenia liczby danych
            # rr2= exclude_max(column_name, rr)
            # last_position_2 = take_max(column_name, rr2)['position'].values[0]

            r['raceId'] = row['raceId']
            r['driverId'] = row['driverId']
            r['last_circuit_position'] = last_position_1
            # r['last_circuit_position2'] = last_position_2
            # r['last_circuit_position3'] = last_position_3

            temp_df = temp_df.append(r, ignore_index=True)
        except Exception:
            pass
    if not temp_df.empty:
        new_df = pd.merge(train_df, temp_df, how='inner', left_on=[ 'raceId','driverId'], right_on=[ 'raceId','driverId'])
        return new_df
    else:
        return temp_df


def take_max(column_n, rr2):

    return rr2.loc[rr2[column_n] == rr2[column_n].max()]


def exclude_max(column_name, rr):
    return rr.loc[rr[column_name] != rr[column_name].max()]

def add_driver_fails(results, train_df):
    driver_all_fails_table = count_driver_fails(results)
    train_df = train_df.merge(driver_all_fails_table, how="outer", left_on='driverId', right_on='driverId',
                              suffixes=('_left', '_right'))
    print(driver_all_fails_table)
    return train_df

def append_driver_constructor_points(df):
    """

    """
    df_merged_standings = create_merged_standings_df()
    for index, row in tqdm(df.iterrows()):
        if TAKE_ONLY_FIRST_100 and row['raceId'] > TAKE_ONLY_FIRTS_ROWS:
            continue
        driverId = row["driverId"]
        ts = row["datetime"]
        year = ts.year
        points_df = get_driver_last_season_points(driverId, year, df_merged_standings)
        df.at[index, 'ls_driver_points'] = int(points_df["driver_points"])
        df.at[index, 'ls_constructor_points'] = int(points_df["constructor_points"])


    return df


def get_driver_last_season_points(driverId, year, df):
    """
        Dla kazdego elemntu w glownej train_df wywołac te funkcje.
        Niektorzy kierowcy z lat 1950-19?? maja 2 konstruktorow.
        :return punkty kierowcy i konstruktora uzyskane w poprzednim sezonie
    """
    year -= 1
    # result = df.loc[df['raceId'] == raceId, ['year', 'driverId', 'driver_points', 'constructorId', 'constructor_points']]
    result = df.loc[df['year'] == year, ['driverId', 'driver_points', 'constructorId', 'constructor_points']]
    result = result.loc[df['driverId'] == driverId, ['driver_points', 'constructor_points']]
    result = result.reset_index()
    result.rename({'index': 'driverId'}, axis=1, inplace=True)
    result = result.sort_values('driver_points', ascending=False).head(1)

    if result.empty:
        result.loc[len(df.index)] = [driverId, 5, 10]

    return result


def create_merged_standings_df():
    driver_standings_df = get_driver_standings_df()
    driver_standings_df.rename({'points': 'driver_points'}, axis=1, inplace=True)
    constructor_standings_df = get_constructor_standings_df()
    constructor_standings_df.rename({'points': 'constructor_points'}, axis=1, inplace=True)
    races_df = get_races_df()
    results_df = get_results_df()
    results_df = results_df[['raceId', 'driverId', 'constructorId']]

    merged_df = races_df[['raceId', 'year']]
    merged_df = merged_df.merge(driver_standings_df[['raceId', 'driverId', 'driver_points']],
                                how="outer", left_on='raceId', right_on='raceId',
                                suffixes=('_left', '_right'))

    merged_df = merged_df.merge(results_df,
                                how="inner", left_on=['raceId', 'driverId'], right_on=['raceId', 'driverId'],
                                suffixes=('_left', '_right'))

    merged_df = merged_df.merge(constructor_standings_df[['raceId', 'constructorId', 'constructor_points']],
                                how="left", left_on=['raceId', 'constructorId'], right_on=['raceId', 'constructorId'],
                                suffixes=('_left', '_right'))

    merged_df = pd.pivot_table(merged_df[merged_df.year >= (YEAR_FROM_CFG -1)], index=['year', 'driverId', 'constructorId'],
                               values=['driver_points', 'constructor_points'], aggfunc='max')

    merged_df.columns.name = None  # remove categories
    merged_df = merged_df.reset_index()

    return merged_df


def get_races_df():
    races_df = pd.read_csv("./f1db_csv/races.csv", delimiter=",")
    races_df = races_df[races_df['year'] >= YEAR_FROM_CFG]


    return races_df


def get_results_df():
    results_df = pd.read_csv("./f1db_csv/results.csv", delimiter=",")

    return results_df


def get_drivers_df():
    drivers_df = pd.read_csv("./f1db_csv/drivers.csv", delimiter=",")

    return drivers_df


def convert_dob_to_age(train_df):
    train_df['dob'] = train_df['dob'].astype(str)
    train_df.dob = pd.to_datetime(train_df.dob, format='%Y-%m-%d')
    train_df['dob'] = train_df['dob'].apply(calculate_age)
    train_df.rename({'dob': 'age'}, axis=1, inplace=True)

    pass


def get_driver_standings_df():
    driver_standings_df = pd.read_csv("./f1db_csv/driver_standings.csv", delimiter=",")

    return driver_standings_df


def get_constructor_standings_df():
    constructor_standings_df = pd.read_csv("./f1db_csv/constructor_standings.csv", delimiter=",")

    return constructor_standings_df


def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def test_data_preparation():

    df_races = pd.read_csv('./f1db_csv/test/racestest.csv')
    df_results = pd.read_csv('./f1db_csv/test/resultstest.csv')

    #train_df = prepare_for_train2(df_races, df_results)
    #date time filtering [(train_df['date']>datetime.date(2009,1,1)) & (train_df['date']<datetime.date(2010,1,1))]
    #print(train_df.groupby(['raceId'])['driverId'].count())

#
# df  = prepare_for_train()
# print(df.head(10))