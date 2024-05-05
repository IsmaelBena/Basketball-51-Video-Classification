import torch.nn as nn
import torch

class CNNModel(nn.Module):
    def __init__(self, device):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv3d(in_channels = 3, out_channels = 24, kernel_size = (5, 3, 3), stride = 3, device = device)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.dropout1 = nn.Dropout3d(0.2)
        self.batchnorm1 = nn.BatchNorm3d(24, device = device)

        self.conv2 = nn.Conv3d(in_channels = 24, out_channels = 48, kernel_size = (3, 2, 2), stride = 2, device = device)
        self.act1 = nn.ReLU()
        self.pool2 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.dropout2 = nn.Dropout3d(0.2)
        self.batchnorm2 = nn.BatchNorm3d(48, device = device)

        self.conv3 = nn.Conv3d(in_channels = 48, out_channels = 64, kernel_size = (3, 2, 2), stride = 2, device = device)
        self.act1 = nn.ReLU()
        self.pool3 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.dropout3 = nn.Dropout3d(0.2)
        self.batchnorm3 = nn.BatchNorm3d(64, device = device)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.dense1 = nn.Linear(256, 512, device=device)
        self.act1 = nn.ReLU()
        self.dense2 = nn.Linear(512, 256, device=device)
        self.act2 = nn.ReLU()
        self.dense3 = nn.Linear(256, 8, device=device)

    def forward(self, input):

        x = self.conv1(input)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.batchnorm3(x)

        x = self.flatten(x)

        x = self.act1(self.dense1(x))
        x = self.act2(self.dense2(x))
        x = self.dense3(x)

        logits = x.clone().detach().requires_grad_(True)

        return logits
    
    