import torch.nn as nn
import torch

class BaselineModel(nn.Module):
    def __init__(self, device):
        super(BaselineModel, self).__init__()

        self.conv1 = nn.Conv3d(in_channels = 3, out_channels = 24, kernel_size = (5, 3, 3), stride = 3, device = device) # output shape: d=29, h=21, w=21
        self.pool1 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.dropout1 = nn.Dropout3d(0.2)
        self.batchnorm1 = nn.BatchNorm3d(24, device = device)

        self.conv2 = nn.Conv3d(in_channels = 24, out_channels = 48, kernel_size = (2, 3, 3), stride = 2, device = device) # output shape: d=14, h=10, w=10
        self.pool2 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.dropout2 = nn.Dropout3d(0.2)
        self.batchnorm2 = nn.BatchNorm3d(48, device = device)

        self.conv3 = nn.Conv3d(in_channels = 48, out_channels = 64, kernel_size = (2, 3, 3), stride = 2, device = device) # output shape: d=19, h=25, w=33
        self.pool3 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.dropout3 = nn.Dropout3d(0.2)
        self.batchnorm3 = nn.BatchNorm3d(64, device = device)

        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)

        self.lstm = nn.LSTM(320, 256, device = device)
        self.linear = nn.Linear(256, 8, device = device)

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

        x, _ = self.lstm(x.view(len(input), 1, -1))
        
        x = self.linear(x.view(len(input), -1))

        logits = x.clone().detach().requires_grad_(True)

        return logits
    
    