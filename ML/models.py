import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from ML.ensemble import MyRandomForestQuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from ML.preprocessing import preproc_linreg, preproc_filled_discrete
from utils.utils import no_outliers, sub_pred_to_pred
from utils.get_data import load
from utils.settings import VAL_DATA_PATH, TARGET_COL, SUB_PRED_VAR


# Base model
class BaseModel:
    def __init__(self, data, sub_pred_var):
        self.data = data
        self.sub_pred_var = sub_pred_var
        if self.sub_pred_var is None:
            self.sub_pred_var = SUB_PRED_VAR
        if self.sub_pred_var is None:
            self.sub_pred_var = TARGET_COL
        self.val_split = VAL_DATA_PATH is None

        if self.val_split:
            data_train, data_val = train_test_split(self.data, test_size=0.33)
            data_train = no_outliers(data_train)
            self.fit(data_train)
            self.data_val = data_val
        else:
            self.fit(self.data)
            self.data_val = load(VAL_DATA_PATH)

    def fit(self, data):
        preproc = self.preprocessing(data)
        target = data[self.sub_pred_var]
        self.rootmodel.fit(preproc, target)

    def predict(self, X):
        preproc = self.preprocessing(X)
        pred = pd.DataFrame(self.rootmodel.predict(preproc), index=X.index)

        if self.sub_pred_var != TARGET_COL:
            pred = sub_pred_to_pred(pred, X, self.sub_pred_var)

        return pred

    def grid_search_cv(self, parameters, scorer, cv):
        gs_data = None
        if self.val_split:
            gs_data = self.data
        else:
            gs_data = pd.concat([self.data, self.data_val])
            train_index = np.array(range(len(self.data)))
            val_index = np.array(range(len(self.data), len(self.data) + len(self.data_val)))
            cv = zip([train_index], [val_index])

        gs = GridSearchCV(self.rootmodel.__class__(), parameters, cv=cv, scoring=scorer)
        preproc = self.preprocessing(gs_data)
        target = gs_data[self.sub_pred_var]
        gs.fit(preproc, target)

        print('Grid scores on development set:\n')
        means = gs.cv_results_['mean_test_score']
        scores = list(zip(means, gs.cv_results_['params']))
        scores = reversed(sorted(scores, key=lambda x: x[0]))
        for mean, params in scores:
            print('{:.3f} - {}'.format(mean, params))
        print('\nBest parameters set found on development set:\n')
        print(gs.best_params_)


class Regressor(BaseModel):
    def __init__(self, data, sub_pred_var):
        super().__init__(data, sub_pred_var=sub_pred_var)

    def grid_search_cv(self, parameters, scorer='explained_variance', cv=5):
        super().grid_search_cv(parameters, scorer=scorer, cv=cv)


class Classifier(BaseModel):
    def __init__(self, data, sub_pred_var):
        super().__init__(data, sub_pred_var=sub_pred_var)

    def grid_search_cv(self, parameters, scorer='f1_macro', cv=5):
        super().grid_search_cv(parameters, scorer=scorer, cv=cv)


# Linear regressors
class LinReg(Regressor):
    def __init__(self, data, formula, sub_pred_var=None):
        self.formula = formula
        self.preprocessing = preproc_linreg
        super().__init__(data, sub_pred_var=sub_pred_var)

    def fit(self, data):
        Xy = self.preprocessing(data)
        Xy[self.sub_pred_var] = data[self.sub_pred_var]
        self.rootmodel = smf.ols(self.sub_pred_var + ' ~ ' + self.formula, data=Xy).fit()

    def grid_search_cv(self, params, scorer='explained_variance', cv=5):
        raise Warning('Grid search is not available for OLS models')


class LinReg1(LinReg):
    def __init__(self, data, sub_pred_var=None):
        self.name = 'Basic linear regression'
        formula = 'HEATING_MODE:REGION + CARETAKER:REGION + ELEVATOR:REGION -1'
        super().__init__(data, formula, sub_pred_var=sub_pred_var)


class LinReg2(LinReg):
    def __init__(self, data, sub_pred_var=None):
        self.name = 'Linear regression'
        formula = 'HEATING_MODE:REGION + CARETAKER:REGION + ELEVATOR + PARKING + ' \
                  'SURFACE -1'
        super().__init__(data, formula, sub_pred_var=sub_pred_var)


class RidgeReg(Regressor):
    def __init__(self, data, sub_pred_var=None):
        self.rootmodel = Ridge()
        self.preprocessing = preproc_filled_discrete
        self.name = 'Ridge regression'
        super().__init__(data, sub_pred_var=sub_pred_var)


class EN(Regressor):
    def __init__(self, data, sub_pred_var=None):
        self.rootmodel = ElasticNet()
        self.preprocessing = preproc_filled_discrete
        self.name = 'Elastic net'
        super().__init__(data, sub_pred_var=sub_pred_var)


class HuberReg(Regressor):
    def __init__(self, data, sub_pred_var=None):
        self.rootmodel = HuberRegressor()
        self.preprocessing = preproc_filled_discrete
        self.name = 'Huber regression'
        super().__init__(data, sub_pred_var=sub_pred_var)


# Ensemble regressors
class RFR(Regressor):
    def __init__(self, data, sub_pred_var=None):
        self.rootmodel = RandomForestRegressor(n_estimators=100, min_samples_leaf=2)
        self.preprocessing = preproc_filled_discrete
        self.name = 'Random forest'
        super().__init__(data, sub_pred_var=sub_pred_var)

    def get_feature_importances(self):
        feature_importances = list(zip([round(var_imp, 3) for var_imp in self.rootmodel.feature_importances_],
                                       self.preprocessing(self.data[:1]).columns))
        for var_imp in reversed(sorted(feature_importances)):
            print(var_imp)


class AdaBoostReg(Regressor):
    def __init__(self, data, sub_pred_var=None):
        self.rootmodel = AdaBoostRegressor()
        self.preprocessing = preproc_filled_discrete
        self.name = 'AdaBoost'
        super().__init__(data, sub_pred_var=sub_pred_var)


class GBR(Regressor):
    def __init__(self, data, sub_pred_var=None):
        self.rootmodel = GradientBoostingRegressor()
        self.preprocessing = preproc_filled_discrete
        self.name = 'Gradient boosting'
        super().__init__(data, sub_pred_var=sub_pred_var)


class RFQR(Regressor):
    def __init__(self, data, sub_pred_var=None):
        self.rootmodel = MyRandomForestQuantileRegressor(n_estimators=1, min_samples_leaf=320)
        self.preprocessing = preproc_filled_discrete
        self.name = 'Random forest quantiles'
        super().__init__(data, sub_pred_var=sub_pred_var)

    def predict(self, X, quantiles=None):
        preproc = self.preprocessing(X)
        pred = self.rootmodel.predict(preproc, quantiles=quantiles)

        if self.sub_pred_var != TARGET_COL:
            pred = sub_pred_to_pred(pred, X, self.sub_pred_var)

        pred = pred.applymap(lambda x: int(round(x, 0)))
        return pred

    def get_feature_importances(self):
        feature_importances = list(zip([round(var_imp, 3) for var_imp in self.rootmodel.feature_importances_],
                                       self.preprocessing(self.data[:1]).columns))
        for var_imp in reversed(sorted(feature_importances)):
            print(var_imp)
