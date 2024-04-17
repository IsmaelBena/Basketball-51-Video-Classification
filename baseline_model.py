import torch.nn as nn
import torch

class BaselineModel(nn.Module):
    def __init__(self, device):
        super(BaselineModel, self).__init__()

        # Conv Layers (videos arre 320x240)

        # Initial kernel dims: stride=9, d=18 h=24, w=32

        self.conv1 = nn.Conv3d(in_channels = 3, out_channels = 4, kernel_size = (18, 24, 32), stride = 9, device = device) # output shape: d=19, h=25, w=33
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d((2, 3, 3), stride = 1)


        self.conv2 = nn.Conv3d(in_channels = 4, out_channels = 5, kernel_size = (4, 5, 7), stride = 2, device = device) # output shape: d=7, h=11, w=14
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d((1, 3, 4), stride = 1)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        # LSTM
        self.lstm = nn.Sequential(
            nn.LSTM(3200, 256),
            nn.Linear(256, 8)
        )

        

    def forward(self, input):
        print(f'input: {input.shape}')

        x = self.act1(self.conv1(input))
        print(f'act1: {x.shape}')
        x = self.pool1(x)
        print(f'pool1: {x.shape}')

        x = self.act2(self.conv2(x))
        print(f'act2: {x.shape}')
        x = self.pool2(x)
        print(f'pool2: {x.shape}')

        x = self.flatten(x)
        print(f'flattened: {x.shape}')

        x, _ = self.lstm[0](x.view(len(input), 1, -1))
        # print(f'lstm1: {x.shape}')
        x = self.lstm[1](x.view(len(input), -1))
        #print(f'lstm2: {x.shape}')

        y_pred = nn.functional.log_softmax(x, dim=1)

        y_pred = torch.tensor(y_pred, dtype = torch.long)

        return y_pred
    
    