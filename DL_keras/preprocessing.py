import pandas as pd
from datetime import datetime
import os
import pickle
from sklearn.model_selection import train_test_split

from utils.get_data import load
from utils.settings import DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, \
    TRAIN_PREPROC_PATH, TEST_PREPROC_PATH, VAL_PREPROC_PATH, MEDIANS_PATH, STDS_PATH

outlier_threshold = 30
dummy_vals = {
    'DEPT_CODE': [None, '01', '02', '03', '04', '06', '07', '08', '09', '93', '11', '12', '67', '13', '14', '15', '17',
                  '16', '18', '19', '2A', '21', '22', '23', '79', '24', '25', '26', '91', '28', '27', '29', '30', '32',
                  '33', '97', '2B', '31', '43', '52', '05', '70', '74', '65', '87', '68', '92', '34', '35', '37', '36',
                  '38', '39', '40', '44', '42', '41', '45', '47', '46', '48', '49', '50', '51', '53', '54', '55', '56',
                  '57', '58', '59', '60', '61', '75', '62', '63', '64', '66', '69', '71', '72', '73', '77', '76', '80',
                  '82', '81', '90', '94', '95', '83', '84', '85', '88', '89', '78', '10', '20', '86', '2b', '974',
                  '972'], 'HEATING_ZONE': ['H1c', 'H1a', 'H2d', 'H3', 'H1b', 'H2c', 'H2b', 'H2a']}


def normalize(df, medians=None, stds=None, with_target=False):
    if medians is None:
        with open(MEDIANS_PATH, 'rb') as file:
            medians = pickle.load(file)
    if stds is None:
        with open(STDS_PATH, 'rb') as file:
            stds = pickle.load(file)

    for c in df.columns:
        if c != 'CONDOMINIUM_EXPENSES' or with_target:
            df[c] = df[c] - medians[c]
            if stds[c] != 0:
                df[c] = df[c] / stds[c]

    return df


def denormalize(y, target_cols):
    medians = None
    stds = None
    with open(MEDIANS_PATH, 'rb') as file:
        medians = pickle.load(file)
    with open(STDS_PATH, 'rb') as file:
        stds = pickle.load(file)

    cols_medians = [medians[c] for c in target_cols]
    cols_stds = [stds[c] for c in target_cols]

    return y * cols_stds + cols_medians


def preprocessing(data, first_stage=False, normalize_data=True):
    cols = ['CONSTRUCTION_YEAR', 'ELEVATOR', 'FLOOR', 'FLOOR_COUNT', 'HEATING_MODE', 'LOT_COUNT', 'PRICE', 'SURFACE', 'CARETAKER',
            'CONDOMINIUM_EXPENSES']
    dummy_cols = ['DEPT_CODE', 'HEATING_ZONE']

    drop_cols = [c for c in data.columns if c not in cols and c not in dummy_cols]
    data.drop(drop_cols, axis=1, inplace=True)  # drop columns

    for col in dummy_cols:  # add dummy variables
        i = 0
        for val in dummy_vals[col]:
            if i > 0:
                data[col + '_' + val] = data[col] == val
            i += 1
    data.drop(dummy_cols, axis=1, inplace=True)

    if not first_stage:
        if normalize_data:
            data = normalize(data)  # normalize
        for c in data.columns:
            if c != 'CONDOMINIUM_EXPENSES':
                data[c] = data[c].mul(data[c].abs() < outlier_threshold)
        data.fillna(0.0, inplace=True)  # fillna
    elif normalize_data:
        medians = data.median()
        stds = data.std()
        data = normalize(data, medians=medians, stds=stds, with_target=True)

    return data.astype('float')


def generate_preproc_if_not_done(split=True, test_size=0.3, low_memory=True, regenerate=False):
    if not os.path.isfile(TRAIN_PREPROC_PATH) or not os.path.isfile(VAL_PREPROC_PATH) or regenerate:
        print('Generating preprocessed datasets')
        train_data, val_data = None, None
        if split:
            data = load(DATA_PATH, low_memory=low_memory)
            train_data, val_data = train_test_split(data, test_size=test_size)
        else:
            train_data = load(DATA_PATH, low_memory=low_memory)
            val_data = load(VAL_DATA_PATH, low_memory=low_memory)

        data_first_stage = train_data.copy()
        preproc_first_stage = None
        removing_outliers = True
        outliers_count = 0
        n_rows = train_data.shape[0]
        while removing_outliers:  # remove outliers for normalization
            preproc_first_stage = preprocessing(data_first_stage.copy(), first_stage=True)
            cond = ((preproc_first_stage > outlier_threshold) |
                    (preproc_first_stage < -outlier_threshold)).any(axis=1)
            data_first_stage = data_first_stage[~cond]
            outliers_count += cond.sum()
            if not cond.any():
                print('Removed {} outliers among {} rows for normalization'.format(outliers_count, n_rows))
                removing_outliers = False

        preproc_first_stage = preprocessing(data_first_stage.copy(), first_stage=True, normalize_data=False)
        medians = dict(preproc_first_stage.median())
        stds = dict(preproc_first_stage.std())
        with open(MEDIANS_PATH, 'wb') as file:
            pickle.dump(medians, file)
        with open(STDS_PATH, 'wb') as file:
            pickle.dump(stds, file)

        train_preproc = preprocessing(train_data.copy())
        val_preproc = preprocessing(val_data.copy())
        train_preproc.to_csv(TRAIN_PREPROC_PATH)
        val_preproc.to_csv(VAL_PREPROC_PATH)

        if TEST_DATA_PATH is not None:
            test_data = load(TEST_DATA_PATH, low_memory=low_memory)
            test_preproc = preprocessing(test_data.copy())
            test_preproc.to_csv(TEST_PREPROC_PATH)
