import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class GestureDataset(Dataset):
    def __init__(self, path):

        # sensor-wise normalization
        # self.df = (df - df.min(axis=0))/(df.max(axis=0) - df.min(axis=0))
        self.df = pd.read_csv(path, index_col='id').to_numpy()
        self.sensors = self.df[:, :-1]
        self.targets = self.df[:, -1]

        # normalize
        # self.sensors = np.transpose(self.sensors)
        # self.sensors = (self.sensors - self.sensors.min(axis=0)) / (self.sensors.max(axis=0) - self.sensors.min(axis=0))
        # self.sensors = (self.sensors - self.sensors.mean(axis=0)) / self.sensors.std(axis=0)
        # self.sensors = np.transpose(self.sensors)
        # self.sensors = (self.sensors - self.sensors.min(axis=0)) / (self.sensors.max(axis=0) - self.sensors.min(axis=0))
        # self.sensors = (self.sensors - self.sensors.mean(axis=0)) / self.sensors.std(axis=0)

        self.len = len(self.df)

    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        x = self.sensors[index]
        target = int(self.targets[index])
        return x, target


if __name__ == '__main__':
    trainset = GestureDataset(path='./hand_gesture_data/train.csv')

    for data in trainset:
        x, target = data
        print(x)
        print(target)
        break
