from keras import models
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import optimizers
from keras import initializers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle

from DL_keras.preprocessing import generate_preproc_if_not_done, denormalize
from utils.get_data import load
from utils.settings import TRAIN_PREPROC_PATH, VAL_PREPROC_PATH
from utils.utils import plot_distrib

generate_preproc_if_not_done(split=True, low_memory=False)

# load data
data_train = load(TRAIN_PREPROC_PATH, low_memory=False)
data_val = load(VAL_PREPROC_PATH, low_memory=False)

# initialize net
def build_model():
    model = models.Sequential()
    model.add(Dense(64, input_shape=(117,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['mae'])

    return model

# train net
rf_pred_train = None
with open('data/rf_pred_train.pkl', 'rb') as file:
    rf_pred_train = pickle.load(file)
rf_pred_val = None
with open('data/rf_pred_val.pkl', 'rb') as file:
    rf_pred_val = pickle.load(file)
gb_pred_train = None
with open('data/gb_pred_train.pkl', 'rb') as file:
    gb_pred_train = pickle.load(file)
gb_pred_val = None
with open('data/gb_pred_val.pkl', 'rb') as file:
    gb_pred_val = pickle.load(file)

X_train = data_train.drop(['CONDOMINIUM_EXPENSES'], axis=1)
X_train_nn = X_train.copy()
# X_train_nn['RF'] = rf_pred_train
# X_train_nn['GB'] = gb_pred_train
y_train = data_train.CONDOMINIUM_EXPENSES
X_val = data_val.drop('CONDOMINIUM_EXPENSES', axis=1)
X_val_nn = X_val.copy()
# X_val_nn['RF'] = rf_pred_val
# X_val_nn['GB'] = gb_pred_val
y_val = data_val.CONDOMINIUM_EXPENSES

model = build_model()
history = model.fit(X_train_nn, y_train, batch_size=128, epochs=5, validation_data=(X_val_nn, y_val))

# Test
loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mean_absolute_error']
val_mae = history.history['val_mean_absolute_error']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, mae, c='orange', label='Training MAE')
plt.plot(epochs, val_mae, c='blue', label='Validation MAE')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

nn_pred = np.ravel(model.predict(X_val_nn))
nn_error = (nn_pred - y_val).abs()

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_error = (rf_pred - y_val).abs()

gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_val)
gb_error = (gb_pred - y_val).abs()

plot_distrib(nn_error, name='NN')
plot_distrib(rf_error, name='RF')
plot_distrib(gb_error, name='GB', ylim=(0, 2000))
plt.show()

# mean_absolute_error: 421.2891 224.8918
# val_mean_absolute_error: 419.2138 408.3046

# rf : 157
# rf : 396
