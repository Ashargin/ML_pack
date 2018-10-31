import pandas as pd
import matplotlib.pyplot as plt
import time

from work.get_data import load_no_nan
from work.models import LinReg1, LinReg2, LinReg3, LinReg4, RidgeReg, EN, HuberReg, RFR, AdaBoostReg, GBR, RFQR
from work.model_error import ModelError
from work.utils import sim_model, get_rfqr_interval

## Data/settings
data = load_no_nan('all-ads-2018.05.01_prepared',
                   columns=['EXPENSES_M2', 'HEATING_MODE', 'CONSTRUCTION_YEAR', 'FLOOR', 'FLOOR_COUNT', 'LOT_COUNT',
                            'PRICE'])
data.reset_index(inplace=True)  # car CARETAKER est en index

data_full = load_no_nan('all-ads-2018.05.01_prepared', columns=['EXPENSES_M2'])
data_full.reset_index(inplace=True)

data_temp = data_full[
    data_full[['HEATING_MODE', 'CONSTRUCTION_YEAR', 'FLOOR', 'FLOOR_COUNT', 'LOT_COUNT', 'PRICE']].isna().any(axis=1)]
data_custom = pd.concat([data, data_temp.sample(frac=data.shape[0] / data_temp.shape[0])])

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
sim_model(RFQR, data[:1000], n_sim=n_sim)
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
rfqr = RFQR(data_full)
toy_data = data[:10]
quantile_preds = rfqr.predict(toy_data, quantiles=[0.05, 0.15, 0.375, 0.625, 0.85, 0.95])

intervals_width, intervals_rel_width = get_rfqr_interval(data_custom, quantiles=[0.05, 0.15, 0.375, 0.625, 0.85, 0.95],
                                                         test_sample=1000, n_sim=100)

rfqr = RFQR(data_custom)
time_start = time.time()
for i in range(100):
    pred = rfqr.predict(data[:1], quantiles=[0.05, 0.15, 0.375, 0.625, 0.85, 0.95])
time_lapse = (time.time() - time_start) / 100
