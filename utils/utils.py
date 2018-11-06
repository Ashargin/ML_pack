import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from utils.get_data import load_no_nan


def check_na(X, show_all=False):
    na_columns = []
    for column in X.columns:
        na_rows = X[column][X[column].isna()]
        if len(na_rows) > 0:
            na_columns.append(column)
    if not na_columns:
        print('No missing values')
    else:
        print('Missing values in columns :')
        for var in na_columns:
            print(var)
        if show_all:
            print('\nColumns without missing values :')
            for var in X.columns:
                if var not in na_columns:
                    print(var)


def check_na_base(name, base_columns, show_all=True):
    check_na(load_no_nan(name, columns=base_columns), show_all=show_all)


def no_outliers(data):
    data_no_outliers = data[((data.CONSTRUCTION_YEAR >= 1000)
                             & (data.CONSTRUCTION_YEAR <= 2018)
                             | (data.CONSTRUCTION_YEAR.isna()))
                            & ((data.FLOOR >= 0)
                               & (data.FLOOR < 50)
                               | (data.FLOOR.isna()))
                            & ((data.LOT_COUNT < 1000)
                               | (data.LOT_COUNT.isna()))
                            ]
    return data_no_outliers


def sub_pred_to_pred(y_pred, X, sub_pred_var):
    if 'LOG' in sub_pred_var:
        y_pred = np.exp(y_pred)
    if 'M2' in sub_pred_var:
        y_pred = y_pred.multiply(X.SURFACE, axis=0)
    return y_pred


def plot_2D(X, Y, title='', xlabel='', ylabel=''):
    plt.figure()
    plt.scatter(X, Y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_3D(X, Y, Z, title='', xlabel='', ylabel=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def cluster_plot_2D(x, assign, k, title='', xlabel='', ylabel=''):
    n = len(x)
    plt.figure()
    colors = []
    if k <= 11:
        colors = np.array(['blue', 'red', 'forestgreen', 'gold', 'darkviolet', 'darkorange',
                           'gray', 'black', 'deepskyblue', 'firebrick', 'lightgreen'])
    else:
        cmap = plt.cm.get_cmap('nipy_spectral', k)
        colors = [cmap(i) for i in range(k)]
    plot_colors = []
    for i in range(n):
        plot_colors.append(colors[assign[i]])
    X = np.array([x[i][0] for i in range(n)])
    Y = np.array([x[i][1] for i in range(n)])
    plt.scatter(X, Y, c=plot_colors)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def cluster_plot_3D(x, assign, k, title='', xlabel='', ylabel=''):
    n = len(x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = []
    if k <= 11:
        colors = np.array(['blue', 'red', 'forestgreen', 'gold', 'darkviolet', 'darkorange',
                           'gray', 'black', 'deepskyblue', 'firebrick', 'lightgreen'])
    else:
        cmap = plt.cm.get_cmap('nipy_spectral', k)
        colors = [cmap(i) for i in range(k)]
    plot_colors = []
    for i in range(n):
        plot_colors.append(colors[assign[i]])
    X = np.array([x[i][0] for i in range(n)])
    Y = np.array([x[i][1] for i in range(n)])
    Z = np.array([x[i][2] for i in range(n)])
    ax.scatter(X, Y, Z, c=plot_colors)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
