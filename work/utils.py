import pandas as pd

from work.getData import load_no_nan
from work.ModelError import ModelError


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


def sim_model(Model, data, sub_pred_var=None, n_sim=1):
    mean_err = 0
    median_err = 0
    mean_err_rel = 0
    median_err_rel = 0

    if sub_pred_var is None:
        model = Model(data)
    else:
        model = Model(data, sub_pred_var=sub_pred_var)
    for i in range(n_sim):
        if i > 0:
            if sub_pred_var is None:
                model = Model(data)
            else:
                model = Model(data, sub_pred_var=sub_pred_var)

        model_error = ModelError(model)
        error = model_error.get_error()
        error_rel = model_error.get_error(relative=True)

        mean_err += error.mean() / n_sim
        median_err += error.median() / n_sim
        mean_err_rel += error_rel.mean() / n_sim
        median_err_rel += error_rel.median() / n_sim

    print('================= ' + model.name + ' =================\n' +
          'Données de test :\n' +
          '\t' + str(model.data_val.shape[0]) + ' lignes.\n' +
          '\t' + str(len(data.REGION.unique())) + ' régions.\n\n' +
          'En moyenne pour une prédiction :\n' +
          '\t' + str(round(model.data_val.CONDOMINIUM_EXPENSES.mean(), 2)) + ' pour CONDOMINIUM_EXPENSES.\n' +
          '\t' + str(round(mean_err, 2)) + " d'erreur en valeur absolue.\n" +
          '\t' + str(round(mean_err_rel, 3)) + " d'erreur relative en valeur absolue.\n\n" +
          'Médiane pour les prédictions :\n' +
          '\t' + str(round(model.data_val.CONDOMINIUM_EXPENSES.median(), 2)) + ' pour CONDOMINIUM_EXPENSES.\n' +
          '\t' + str(round(median_err, 2)) + " d'erreur en valeur absolue.\n" +
          '\t' + str(round(median_err_rel, 3)) + " d'erreur relative en valeur absolue.\n" +
          '============================================='
          )


def get_rfqr_interval(data, quantiles, test_sample=10000, interval=None, sub_pred_var=None, n_sim=100):
    if interval is None:
        interval = [0.05, 0.95]
    from notebooks.models import RFQR
    data_temp = load_no_nan('all-ads-2018.05.01_prepared', columns=['EXPENSES_M2'])
    data_temp.reset_index(inplace=True)
    data_test = data_temp.sample(frac=test_sample / data_temp.shape[0])
    dfs_preds = []
    if sub_pred_var is None:
        rfqr = RFQR(data, val_split=False)
    else:
        rfqr = RFQR(data, sub_pred_var=sub_pred_var, val_split=False)

    for i in range(n_sim):
        if i > 0:
            if sub_pred_var is None:
                rfqr = RFQR(data, val_split=False)
            else:
                rfqr = RFQR(data, sub_pred_var=sub_pred_var, val_split=False)

        quantiles_preds = rfqr.predict(data_test, quantiles=quantiles)
        dfs_preds.append(quantiles_preds)
    df_tot = pd.concat(dfs_preds)
    gb_tot = df_tot.groupby(df_tot.index)
    intervals_width = gb_tot.quantile(interval[1]) - gb_tot.quantile(interval[0])
    intervals_rel_width = intervals_width.divide(gb_tot.mean())
    return intervals_width, intervals_rel_width


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
