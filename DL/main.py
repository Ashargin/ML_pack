import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os.path

from DL.preprocessing import generate_preproc_if_not_done, denormalize
from DL.net import LinearNet, TrainDfDataset, TestDfDataset
from DL.utils import RMSELoss
from utils.get_data import load
from utils.settings import TRAIN_PREPROC_PATH, TEST_PREPROC_PATH

generate_preproc_if_not_done(split=True, low_memory=False)

# load data
X_train = load(TRAIN_PREPROC_PATH, low_memory=False)
X_train_set = TrainDfDataset(X_train, target_cols=['CONDOMINIUM_EXPENSES'])
X_train_loader = DataLoader(X_train_set, batch_size=4, shuffle=True)

# initialize net
net = LinearNet([116, 50, 16, 1])

# train net
criterion = nn.L1Loss()  # RMSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

net.start_training(X_train_loader, criterion=criterion, optimizer=optimizer, epochs=0.5)

# 4 problèmes :
# > loss overflow, sometimes nan
# > constant output : parameters?
# initialize weights

X_test = load(TEST_PREPROC_PATH, low_memory=False)
X_test_set = TestDfDataset(X_test, target_cols=['CONDOMINIUM_EXPENSES'])
y_test = denormalize(net.get_eval(X_test_set))
