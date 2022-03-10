import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layer(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Sequential(
            LinearBlock(32, 128),
            # nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            LinearBlock(128, 128),
            # nn.Dropout(0.5)
        )
        # self.linear3 = nn.Sequential(
        #     LinearBlock(128, 128),
        #     # nn.Dropout(0.5)
        # )
        self.linear4 = nn.Sequential(
            nn.Linear(128, 4)
        )
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        # x = self.linear3(x)
        x = self.linear4(x)
        return x


if __name__ == "__main__":
    import os
    import sys
    sys.path.append('.')

    from torch.utils.data import DataLoader
    from dataset import GestureDataset

    model = Model()
    trainset = GestureDataset(os.path.join('hand_gesture_data', 'train.csv'))
    trainloader = DataLoader(trainset, batch_size=64, num_workers=0)

    for xs, targets in trainloader:
        probs = model(xs)
        print(probs)
        break