import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # initialize
        self.model = self.model.to(self.config.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay
        )
    
    def run_epoch(self, epoch, dataset, mode):
        # flag for train mode or not
        is_train = mode == "train"
        
        # performance metrics
        n_correct = 0
        n_incorrect = 0

        # set dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True
        )

        # iterate through dataloader
        losses = []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for it, (xs, targets) in pbar:
            # transfer data into device
            xs = xs.to(self.config.device, dtype=torch.float)
            targets = targets.to(self.config.device, dtype=torch.long)

            # feed model
            with torch.set_grad_enabled(is_train):
                probs = self.model(xs)
                loss = F.cross_entropy(probs.view(-1, probs.size(-1)), targets.view(-1))
                losses.append(loss.item())

            # if you're running in train mode, update parameters
            if is_train:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            # check metrics
            __check = torch.argmax(probs.view(-1, probs.size(-1)), axis=1) == targets.view(-1)
            __correct = torch.sum(__check)
            n_correct += __correct.item()
            n_incorrect += len(__check) - __correct.item()

            # update desciption on progress bar
            loss_value = loss.item() if is_train else float(np.mean(losses))
            acc = n_correct / (n_correct + n_incorrect)
            pbar.set_description(f'epoch {epoch} iter {it}: {mode} loss {loss_value:.5f} acc {acc:.5f}')


if __name__ == "__main__":
    import os
    from torch.utils.data import random_split
    from config import TrainerConfig
    from models.DNNv1 import Model
    from dataset import GestureDataset

    # model 
    model = Model()

    # trainset, valiset, testset
    dataset = GestureDataset(os.path.join('hand_gesture_data', 'train.csv'))
    n_train = int(len(dataset)*0.8)
    n_vali = int(len(dataset)*0.1)
    n_test = len(dataset) - (n_train + n_vali)
    trainset, valiset, testset = random_split(dataset, [n_train, n_vali, n_test])

    # trainer
    trainer = Trainer(model, TrainerConfig())
    for epoch in range(200):
        trainer.run_epoch(epoch, trainset, "train")
        trainer.run_epoch(epoch, valiset, "vali")
        trainer.run_epoch(epoch, testset, "test")


        
