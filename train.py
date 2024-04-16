from dataset import LocalDataset, BasketballVideos
from torch.utils.data import DataLoader
import os

TRAIN_BATCH_SIZE = 64

data = LocalDataset(os.getcwd() + '\\dataset\\').load_dataset()
#print(data.dir)
#print(data.load_dataset())

training_set = BasketballVideos(data)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': True,
                'pin_memory': True
                }

training_loader = DataLoader(training_set, **train_params)

print(training_loader.dataset[0])