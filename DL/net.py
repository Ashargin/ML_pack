import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from termcolor import colored
import math
import time

device_setting = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_setting)
print('Using ' + ('cpu' if device_setting == 'cpu' else
                  'cuda device {}'.format(torch.cuda.get_device_name(0))))


class LinearNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.linear1 = nn.Linear(layers[0], layers[1])
        self.linear2 = nn.Linear(layers[1], layers[2])
        self.linear3 = nn.Linear(layers[2], layers[3])

        self.to(device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def start_training(self, loader, criterion=nn.MSELoss(), optimizer=None, epochs=1, display_rows=10000):
        self.train(True)
        if optimizer is None:
            optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        print('Start training')
        time_start = time.time()
        for epoch in range(math.ceil(epochs)):
            running_loss = 0.0
            i = 0

            for data in loader:
                inputs, targets = data

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                i += loader.batch_size
                if i % display_rows < loader.batch_size:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i, running_loss / display_rows))
                    running_loss = 0.0

                if epoch == math.ceil(epochs) - 1 and i >= (epochs - int(epochs)) * loader.dataset.__len__():
                    break
        print('End training in {:.1f}s'.format(time.time() - time_start))

    def get_eval(self, set):
        self.train(False)

        raise NotImplementedError


class TrainDfDataset(Dataset):
    def __init__(self, data, target_cols):
        self.X = np.array(data.drop(target_cols, axis=1))
        self.y = np.array(data[target_cols])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        input = torch.from_numpy(self.X[idx]).float()
        target = torch.from_numpy(self.y[idx]).float()
        return input.to(device), target.to(device)


class TestDfDataset(Dataset):
    def __init__(self, data, target_cols):
        self.X = np.array(data.drop(target_cols, axis=1))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        input = torch.from_numpy(self.X[idx]).float()
        return input.to(device)
