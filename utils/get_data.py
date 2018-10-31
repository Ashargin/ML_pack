import pickle
import pandas as pd
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

loaded = {}


def load(path, low_memory=True, cache=True):
    if path in loaded and cache:
        return loaded[path].copy()

    df = None
    _, ext = os.path.splitext(path)

    if ext == '.pkl' and os.path.isfile(path):
        df = pickle.load(open(path, 'rb'))
    elif ext == '.csv' and os.path.isfile(path):
        df = pd.read_csv(path, index_col=0, low_memory=low_memory)
        df.drop([c for c in df.columns if 'Unnamed' in c], axis=1, inplace=True)

    if df is None:
        raise ValueError("'" + path + "' can't be found !")
    elif cache:
        loaded[path] = df
        return load(path)
    else:
        return df.copy()


def load_no_nan(path, columns, low_memory=True, cache=True):
    df = load(path, low_memory=low_memory, cache=cache)
    return df[~df[columns].isna().any(axis=1)]


def save(df, path):
    loaded[path] = df
    return pickle.dump(df, open(path, 'wb'))
