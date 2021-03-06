import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os.path

from DL.preprocessing import generate_preproc_if_not_done, denormalize
from DL.net import LinearNet, TrainDfDataset, TestDfDataset
from DL.utils import RMSELoss
from utils.get_data import load
from utils.settings import TRAIN_PREPROC_PATH, VAL_PREPROC_PATH

generate_preproc_if_not_done(split=True, low_memory=False)

# load data
X_train = load(TRAIN_PREPROC_PATH, low_memory=False)
X_train_set = TrainDfDataset(X_train, target_cols=['CONDOMINIUM_EXPENSES'])
X_train_loader = DataLoader(X_train_set, batch_size=4, shuffle=True)

# initialize net
net = LinearNet([116, 50, 16, 1])

# train net
criterion = nn.L1Loss()  # RMSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)

net.start_training(X_train_loader, criterion=criterion, optimizer=optimizer, epochs=0.5)

# 4 problèmes :
# > loss overflow, sometimes nan
# > constant output : parameters?
# initialize weights

X_val = load(VAL_PREPROC_PATH, low_memory=False)
X_val_set = TestDfDataset(X_val, target_cols=['CONDOMINIUM_EXPENSES'])
y_val = denormalize(net.get_eval(X_val_set))
