import pandas as pd
import matplotlib.pyplot as plt
import time

from utils.get_data import load_no_nan
from ML.models import LinReg1, LinReg2, RidgeReg, EN, HuberReg, RFR, AdaBoostReg, GBR, RFQR
from ML.model_error import ModelError
from ML.utils import sim_model
from utils.settings import DATA_PATH, TARGET_COL

## Data/settings
# data = load_no_nan(DATA_PATH, columns=[col1, col2, col3, TARGET_COL])
# data_full = load_no_nan(DATA_PATH, columns=[TARGET_COL])
# data_custom = do_something()

n_sim = 1

## Linear regressors
# First linear regression
sim_model(LinReg1, data_full, n_sim=n_sim)
ModelError(LinReg1(data_full)).plot_error_distrib()

# Second linear regression
sim_model(LinReg2, data_full, n_sim=n_sim)
ModelError(LinReg2(data_full)).plot_error_distrib(color='blue')

# Ridge regression
sim_model(RidgeReg, data_full, n_sim=n_sim)
ModelError(RidgeReg(data_full)).plot_error_distrib(color='darkorange')

# Elastic net
sim_model(EN, data_full, n_sim=n_sim)
ModelError(EN(data_full)).plot_error_distrib(color='black')

# Huber regression
sim_model(HuberReg, data_full, n_sim=n_sim)
ModelError(HuberReg(data_full)).plot_error_distrib(color='darkviolet')

## Ensemble regressors
# Random forest
sim_model(RFR, data_full, n_sim=n_sim)
ModelError(RFR(data_full)).plot_error_distrib(color='forestgreen')

# AdaBoost
sim_model(AdaBoostReg, data_full, n_sim=n_sim)
ModelError(AdaBoostReg(data_full)).plot_error_distrib(color='gray')

# Gradient boosting
sim_model(GBR, data_full, n_sim=n_sim)
ModelError(GBR(data_full)).plot_error_distrib(color='firebrick')

# Random forest quantiles
sim_model(RFQR, data_full, n_sim=n_sim)
ModelError(RFQR(data_full)).plot_error_distrib(color='red')

## Model choice
plt.xlim((0, 1))
plt.ylim((0, 1250))
plt.show()

rf = RFR(data_custom)
parameters = {'n_estimators': [20], 'min_samples_split': [2]}
rf.grid_search_cv(parameters)
rf.get_feature_importances()

## Quantiles prediction
rfqr = RFQR(data_custom)
toy_data = data[:10]
quantile_preds = rfqr.predict(toy_data, quantiles=[0.05, 0.15, 0.375, 0.625, 0.85, 0.95])
