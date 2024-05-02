from dataset import LocalDataset, BasketballVideos
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import yaml
from utils import get_config
import math

from baseline_model import BaselineModel
from grayscale_model import GrayscaleModel
from vgg16_model import VGGModel
from logger import Logger

# ========= PARAMS ==============================================================================================================

training_config = get_config('model')

TRAIN_BATCH_SIZE = training_config['batch_size']
epochs = training_config['epochs']

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

checkpoint_dir = os.path.join(os.getcwd(), training_config["checkpoint_dir"])

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# ===============================================================================================================================

# =================   Base training function that takes all the data at once, too resource hungry.  =============================

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

# ===============================================================================================================================

# =================  Training function that reads the data segments at a time, takes longer but saves on memory.  ===============

def train_in_segments(model, loss_function, optimiser, logger, epochs, data_slider_fraction, save_checkpoint_name='', checkpoint_name='', gray_scale=False, lr=0.5, decay=0.01):
    print("Starting segmented training")
    if checkpoint_name != '':
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_name)))
        print(f'Loaded model checkpoint: {checkpoint_name}')
    
    for epoch in range(epochs):
    
        print(f'Epoch: {epoch}')
        training_losses = []
        eval_losses = []

        model.train()
        start_segment = 0
        segment_num = 0

        # Training loop
        while start_segment < 1:
            segment_num += 1
            print(f'Training segment {segment_num} of {math.ceil(1/data_slider_fraction)}')
            end_segment = start_segment+data_slider_fraction
            if end_segment > 1:
                end_segment = 1

            s_training_data = LocalDataset(os.path.join(os.getcwd(), 'dataset', 'train'), gray_scale=gray_scale, start_segment=start_segment, end_segment=end_segment).load_dataset()
            s_batch = BasketballVideos(s_training_data)
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
                training_losses.append(loss.cpu().detach().numpy())
                print(f'Epoch: {epoch} -- Training batch: {idx+1} of {int(len(s_training_loader.dataset)/TRAIN_BATCH_SIZE)} -- Average Loss: {np.average(training_losses)} -- Progress: {round(((idx+1)*100)/int(len(s_training_loader.dataset)/TRAIN_BATCH_SIZE), 2)}%       ', end='\r', flush=True)
            print('\n')

            start_segment += data_slider_fraction
                
        start_segment = 0
        segment_num = 0

        model.eval()

        pred_true = []

        # Validation loop
        while start_segment < 1:
            segment_num += 1
            print(f'Validation segment {segment_num} of {math.ceil(1/data_slider_fraction)}')
            end_segment = start_segment+data_slider_fraction
            if end_segment > 1:
                end_segment = 1

            s_val_data = LocalDataset(os.path.join(os.getcwd(), 'dataset', 'val'), gray_scale=gray_scale, start_segment=start_segment, end_segment=end_segment).load_dataset()
            s_batch = BasketballVideos(s_val_data)
            s_val_loader = DataLoader(s_batch, **train_params)

            for idx, batch in enumerate(s_val_loader):

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
                # loss.backward()
                # optimiser.step()
                y_pred = torch.softmax(logits, dim = 1)
                y_pred = torch.argmax(y_pred, dim = 1)

                for p_idx, pred in enumerate(y_pred):
                    pred_true.append((pred, y[p_idx]))

                # print(f'\tLoss: {loss}')
                eval_losses.append(loss.cpu().detach().numpy())
                print(f'Epoch: {epoch} -- Validation batch: {idx+1} of {int(len(s_val_loader.dataset)/TRAIN_BATCH_SIZE)} -- Average Loss: {np.average(eval_losses)} -- Progress: {round(((idx+1)*100)/int(len(s_val_loader.dataset)/TRAIN_BATCH_SIZE), 2)}%       ', end='\r', flush=True)
            print('\n')

            start_segment += data_slider_fraction
        
        correct = 0
        for pred in pred_true:
            # print(pred[0], pred[1])
            if pred[0] == pred[1]:
                correct += 1

        if save_checkpoint_name == '':
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'grayscale_s_{data_slider_fraction}'))
        else:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'basemodel_lr_{lr}_decay_{decay}'))

        logger.log({
            'train_loss': np.average(training_losses),
            'eval_loss': np.average(eval_losses),
            'epoch_acc': correct/len(pred_true)
            })

# ===========  Call the functions here ================================================================

# data = LocalDataset(os.path.join(os.getcwd(), 'dataset'), True).load_dataset()
# training_set = BasketballVideos(data)
# training_loader = DataLoader(training_set, **train_params)
# print(len(training_loader.dataset))

# model = BaselineModel(device)
# model = GrayscaleModel(device)
# model = VGGModel(device)

loss_function = torch.nn.CrossEntropyLoss()

lrs = [0.2, 0.1, 0.05]
decays = [0.01, 0.001, 0.0001]
for lr in lrs:
    for decay in decays:
        model = BaselineModel(device)
        optimiser = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = decay)
        wandb_logger = Logger(f"inm705_cw_vgg_model_test_{lr}_{decay}", project='INM705_CW')
        logger = wandb_logger.get_logger()
        model.to(device)
        print(f'LR: {lr} - DECAY: {decay}')
        train_in_segments(model, loss_function, optimiser, logger, epochs, 0.04, gray_scale=False, lr=lr, decay=decay)


# train_model(model, training_loader, loss_function, optimiser, logger, epochs)

# implement segmented loader