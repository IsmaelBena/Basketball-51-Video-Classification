from dataset import LocalDataset, BasketballVideos
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import yaml
from utils import get_config
import math

from baseline_model import BaselineModel
from logger import Logger

# ========= PARAMS ===========

TRAIN_BATCH_SIZE = 2
epochs = 30

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': False,
                'pin_memory': True
                }

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Device: {device}')

# ============================

# data = LocalDataset(os.path.join(os.getcwd(), 'dataset'), 0, 0.02).load_dataset()

checkpoint_dir = os.path.join(os.getcwd(), get_config("model")["checkpoint_dir"])

# training_set = BasketballVideos(data)

# training_loader = DataLoader(training_set, **train_params)

# print(len(training_loader.dataset))

model = BaselineModel(device)
loss_function = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr = 0.5, weight_decay = 0.01)

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
            losses.append(loss.cpu().detach().numpy())
            
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'base_epoch_{epoch}'))

        logger.log({'train_loss': np.average(losses)})

wandb_logger = Logger(f"inm705_cw_initial_model_decay", project='INM705_CW')
logger = wandb_logger.get_logger()

# train_model(model, training_loader, loss_function, optimiser, logger, epochs)

# implement segmented loader

def train_in_segments(model, loss_function, optimiser, logger, epochs, data_slider_fraction, checkpoint_name=''):
    print("Starting segmented training")
    if checkpoint_name != '':
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_name)))
        print(f'Loaded model checkpoint: {checkpoint_name}')

    model.train()
    
    for epoch in range(epochs):
    
        print(f'Epoch: {epoch}')
        losses = []

        start_segment = 0
        segment_num = 0

        while start_segment < 1:
            segment_num += 1
            print(f'Segment {segment_num} of {math.ceil(1/data_slider_fraction)}')
            end_segment = start_segment+data_slider_fraction
            if end_segment > 1:
                end_segment = 1

            s_data = LocalDataset(os.path.join(os.getcwd(), 'dataset'), start_segment=start_segment, end_segment=end_segment).load_dataset()
            s_batch = BasketballVideos(s_data)
            s_training_loader = DataLoader(s_batch, **train_params)

            for idx, batch in enumerate(s_training_loader):

                model.zero_grad()
                
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

                # print(f'\tLoss: {loss}')
                losses.append(loss.cpu().detach().numpy())
                print(f'Epoch: {epoch} -- Batch: {idx} of {int(len(s_training_loader.dataset)/TRAIN_BATCH_SIZE)} -- Average Loss: {np.average(losses)} -- Progress: {round(((idx+1)*100)/int(len(s_training_loader.dataset)/TRAIN_BATCH_SIZE), 2)}%       ', end='\r', flush=True)
            print('\n')

            start_segment += data_slider_fraction
                
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'base_epoch_{epoch}_s_{data_slider_fraction}'))

        logger.log({'train_loss': np.average(losses)})

train_in_segments(model, loss_function, optimiser, logger, epochs, 0.02)