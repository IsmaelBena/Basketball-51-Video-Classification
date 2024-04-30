from dataset import LocalDataset, BasketballVideos
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import yaml
from utils import get_config

from baseline_model import BaselineModel
from logger import Logger

TRAIN_BATCH_SIZE = 24
epochs = 10

data = LocalDataset(os.getcwd() + '\\dataset\\', 1).load_dataset()

checkpoint_dir = os.path.join(os.getcwd(), get_config("model")["checkpoint_dir"])

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Device: {device}')

training_set = BasketballVideos(data)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': False,
                'pin_memory': True
                }

training_loader = DataLoader(training_set, **train_params)

print(len(training_loader.dataset))

model = BaselineModel(device)
loss_function = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr = 0.1, weight_decay = 0.0)

#print(training_loader.dataset[0])
print('hello?')

model.to(device)

def train_model(model, training_loader, loss_function, optimiser, logger, epochs):
    model.train()
    
    for epoch in range(epochs):

        print(f'Epoch: {epoch}')
        losses = []

        for idx, batch in enumerate(training_loader):
            
            model.zero_grad()

            print(f'Epoch: {epoch}, Batch {idx} of {int(len(training_loader.dataset)/TRAIN_BATCH_SIZE)}')
            
            x = batch['videos']

            y = batch['labels']

            #print(f'x Shape: {x.shape}')

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            logits = logits.to(device)

            #y = torch.tensor(y, dtype = torch.long)

            loss = loss_function(logits, y)
            loss.backward()
            optimiser.step()

            print(f'\tLoss: {loss}')
            losses.append(loss)
            
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'base_epoch_{epoch}'))

        logger.log({'train_loss': np.average(losses)})

wandb_logger = Logger(f"inm705_cw_initial_model", project='INM705_CW')
logger = wandb_logger.get_logger()

train_model(model, training_loader, loss_function, optimiser, logger, epochs)