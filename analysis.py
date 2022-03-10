import numpy as np
import matplotlib.pyplot as plt
from dataset import GestureDataset

trainset = GestureDataset(path='./hand_gesture_data/train.csv')

patterns = {
    0: [],  # 569
    1: [],  # 574
    2: [],  # 593
    3: [],  # 599
}

for data in trainset:
    x, target = data
    patterns[target].append(x)

for idx, p in patterns.items():
    mean = np.mean(p, axis=0)
    std = np.std(p, axis=0)
    plt.errorbar(np.arange(len(mean)), mean, std, fmt='-o', capsize=6, elinewidth=1, markeredgewidth=1)

plt.show()