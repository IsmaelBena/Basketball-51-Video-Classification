from dataset import LocalDataset, BasketballVideos
from torch.utils.data import DataLoader
import os
import torch

from baseline_model import BaselineModel

TRAIN_BATCH_SIZE = 10
epochs = 1

data = LocalDataset(os.getcwd() + '\\dataset\\', 0.02).load_dataset()
#print(data.dir)
#print(data.load_dataset())

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

def train_model(model, training_loader, loss_function, optimiser, epochs):
    model.train()
    
    for epoch in range(epochs):

        print(f'Epoch: {epoch}')

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

            print(loss)

train_model(model, training_loader, loss_function, optimiser, epochs)