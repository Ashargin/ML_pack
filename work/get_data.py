import pickle
import pandas as pd
import os

folder = os.path.dirname(os.path.abspath(__file__))  # Current directory
folder_data = os.path.join(folder, '../data/')  # Data directory

loaded = {}


def load(name, cache=True):
    if name in loaded and cache:
        return loaded[name].copy()

    df = None

    path = folder_data + name + '.'
    path_pickle = path + 'pickle'
    path_csv = path + 'csv'

    if os.path.isfile(path_pickle):
        df = pickle.load(open(path_pickle, 'rb'))
    elif os.path.isfile(path_csv):
        df = pd.read_csv(path_csv, index_col=0, low_memory=False)

    if df is None:
        raise ValueError("'" + path_csv + "' can't be found !")
    elif cache:
        loaded[name] = df
        return load(name)
    else:
        return df.copy()


def load_no_nan(name, columns, cache=True):
    df = load(name, cache=cache)
    return df[~df[columns].isna().any(axis=1)]


def save(df, name):
    loaded[name] = df
    path = folder_data + name + '.pickle'
    return pickle.dump(df, open(path, 'wb'))
