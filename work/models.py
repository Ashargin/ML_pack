import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from work.ensemble import MyRandomForestQuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from work.preprocessing import preproc_linreg, preproc_filled_discrete
from work.utils import no_outliers


# Base model
class BaseModel:
    def __init__(self, data, sub_pred_var, val_split):
        self.data = data
        self.sub_pred_var = sub_pred_var

        if val_split:
            data_train, data_val = train_test_split(self.data, test_size=0.33)
            data_train = no_outliers(data_train)
            self.fit(data_train)
            self.data_val = data_val
        else:
            self.fit(self.data)
            self.data_val = self.data

    def fit(self, data):
        preproc = self.preprocessing(data)
        target = data[self.sub_pred_var]
        self.rootmodel.fit(preproc, target)

    def predict(self, X):
        preproc = self.preprocessing(X)
        pred = pd.DataFrame(self.rootmodel.predict(preproc), index=X.index)

        if 'LOG' in self.sub_pred_var:
            pred = np.exp(pred)
        if 'M2' in self.sub_pred_var:
            pred = pred.multiply(X.SURFACE, axis=0)
        return pred

    def grid_search_cv(self, parameters, scorer, cv=5):
        gs = GridSearchCV(self.rootmodel.__class__(), parameters, cv=cv, scoring=scorer)

        preproc = self.preprocessing(self.data)
        target = self.data[self.sub_pred_var]
        gs.fit(preproc, target)

        print('Best parameters set found on development set:\n')
        print(gs.best_params_, '\n')
        print('Grid scores on development set:\n')
        means = gs.cv_results_['mean_test_score']
        stds = gs.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gs.cv_results_['params']):
            print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))


class Regressor(BaseModel):
    def __init__(self, data, sub_pred_var, val_split):
        super().__init__(data, sub_pred_var=sub_pred_var, val_split=val_split)

    def grid_search_cv(self, parameters, scorer='explained_variance', cv=5):
        super().grid_search_cv(parameters, scorer=scorer, cv=cv)


class Classifier(BaseModel):
    def __init__(self, data, sub_pred_var, val_split):
        super().__init__(data, sub_pred_var=sub_pred_var, val_split=val_split)

    def grid_search_cv(self, parameters, scorer='f1_macro', cv=5):
        super().grid_search_cv(parameters, scorer=scorer, cv=cv)


# Linear regressors
class LinReg(Regressor):
    def __init__(self, data, formula, sub_pred_var='LOG_EXPENSES_M2', val_split=True):
        self.formula = formula
        self.preprocessing = preproc_linreg
        super().__init__(data, sub_pred_var=sub_pred_var, val_split=val_split)

    def fit(self, data):
        Xy = self.preprocessing(data)
        Xy[self.sub_pred_var] = data[self.sub_pred_var]
        self.rootmodel = smf.ols(self.sub_pred_var + ' ~ ' + self.formula, data=Xy).fit()

    def grid_search_cv(self, params, scorer='explained_variance', cv=5):
        print('Grid search is not available for OLS models')


class LinReg1(LinReg):
    def __init__(self, data, sub_pred_var='LOG_EXPENSES_M2', val_split=True):
        self.name = 'Basic linear regression'
        formula = 'HEATING_MODE:REGION + CARETAKER:REGION + ELEVATOR:REGION -1'
        super().__init__(data, formula, sub_pred_var=sub_pred_var, val_split=val_split)


class LinReg2(LinReg):
    def __init__(self, data, sub_pred_var='LOG_EXPENSES_M2', val_split=True):
        self.name = 'Linear regression'
        formula = 'HEATING_MODE:REGION + CARETAKER:REGION + ELEVATOR + PARKING + ' \
                  'SURFACE -1'
        super().__init__(data, formula, sub_pred_var=sub_pred_var, val_split=val_split)


class LinReg3(LinReg):
    def __init__(self, data, sub_pred_var='EXPENSES_M2', val_split=False):
        self.name = 'Linear regression'
        formula = 'CARETAKER + HEATING_MODE + ELEVATOR:LOT_COUNT_CAT + CONSTRUCTION_YEAR -1'
        super().__init__(data, formula, sub_pred_var=sub_pred_var, val_split=val_split)


class LinReg4(LinReg):
    def __init__(self, data, sub_pred_var='EXPENSES_M2', val_split=False):
        self.name = 'Linear regression'
        formula = 'CARETAKER:LOT_COUNT_CAT + HEATING_MODE + ELEVATOR + CONSTRUCTION_YEAR + PARKING + HEATING_ZONE -1'
        super().__init__(data, formula, sub_pred_var=sub_pred_var, val_split=val_split)


class RidgeReg(Regressor):
    def __init__(self, data, sub_pred_var='LOG_EXPENSES_M2', val_split=True):
        self.rootmodel = Ridge()
        self.preprocessing = preproc_filled_discrete
        self.name = 'Ridge regression'
        super().__init__(data, sub_pred_var=sub_pred_var, val_split=val_split)


class EN(Regressor):
    def __init__(self, data, sub_pred_var='CONDOMINIUM_EXPENSES', val_split=True):
        self.rootmodel = ElasticNet()
        self.preprocessing = preproc_filled_discrete
        self.name = 'Elastic net'
        super().__init__(data, sub_pred_var=sub_pred_var, val_split=val_split)


class HuberReg(Regressor):
    def __init__(self, data, sub_pred_var='LOG_EXPENSES_M2', val_split=True):
        self.rootmodel = HuberRegressor()
        self.preprocessing = preproc_filled_discrete
        self.name = 'Huber regression'
        super().__init__(data, sub_pred_var=sub_pred_var, val_split=val_split)


# Ensemble regressors
class RFR(Regressor):
    def __init__(self, data, sub_pred_var='LOG_EXPENSES_M2', val_split=True):
        self.rootmodel = RandomForestRegressor(n_estimators=100, min_samples_leaf=2)
        self.preprocessing = preproc_filled_discrete
        self.name = 'Random forest'
        super().__init__(data, sub_pred_var=sub_pred_var, val_split=val_split)

    def get_feature_importances(self):
        feature_importances = list(zip([round(var_imp, 3) for var_imp in self.rootmodel.feature_importances_],
                                       self.preprocessing(self.data_val).columns))
        for var_imp in reversed(sorted(feature_importances)):
            print(var_imp)


class AdaBoostReg(Regressor):
    def __init__(self, data, sub_pred_var='LOG_EXPENSES_M2', val_split=True):
        self.rootmodel = AdaBoostRegressor()
        self.preprocessing = preproc_filled_discrete
        self.name = 'AdaBoost'
        super().__init__(data, sub_pred_var=sub_pred_var, val_split=val_split)


class GBR(Regressor):
    def __init__(self, data, sub_pred_var='LOG_EXPENSES_M2', val_split=True):
        self.rootmodel = GradientBoostingRegressor()
        self.preprocessing = preproc_filled_discrete
        self.name = 'Gradient boosting'
        super().__init__(data, sub_pred_var=sub_pred_var, val_split=val_split)


class RFQR(Regressor):
    def __init__(self, data, sub_pred_var='LOG_EXPENSES_M2', val_split=True):
        self.rootmodel = MyRandomForestQuantileRegressor(n_estimators=1, min_samples_leaf=320)
        self.preprocessing = preproc_filled_discrete
        self.name = 'Random forest quantiles'
        super().__init__(data, sub_pred_var=sub_pred_var, val_split=val_split)

    def predict(self, X, quantiles=None):
        preproc = self.preprocessing(X)
        pred = self.rootmodel.predict(preproc, quantiles=quantiles)

        if 'LOG' in self.sub_pred_var:
            pred = np.exp(pred)
        if 'M2' in self.sub_pred_var:
            pred = pred.multiply(X.SURFACE, axis=0)

        pred = pred.applymap(lambda x: int(round(x, 0)))
        return pred

    def get_feature_importances(self):
        feature_importances = list(zip([round(var_imp, 3) for var_imp in self.rootmodel.feature_importances_],
                                       self.preprocessing(self.data_val).columns))
        for var_imp in reversed(sorted(feature_importances)):
            print(var_imp)
