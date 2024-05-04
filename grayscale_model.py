import torch.nn as nn

class GrayscaleModel(nn.Module):
    def __init__(self, device):
        super(GrayscaleModel, self).__init__()


        self.conv1 = nn.Conv2d(in_channels = 90, out_channels = 110, kernel_size = (3, 3), stride = 3, device = device) # output shape: d=29, h=21, w=21
        self.pool1 = nn.MaxPool2d((3, 3), stride = 1)
        self.dropout1 = nn.Dropout2d(0.2)
        self.batchnorm1 = nn.BatchNorm2d(110, device = device)

        self.conv2 = nn.Conv2d(in_channels = 110, out_channels = 80, kernel_size = (3, 3), stride = 2, device = device) # output shape: d=14, h=10, w=10
        self.pool2 = nn.MaxPool2d((3, 3), stride = 1)
        self.dropout2 = nn.Dropout2d(0.2)
        self.batchnorm2 = nn.BatchNorm2d(80, device = device)

        self.conv3 = nn.Conv2d(in_channels = 80, out_channels = 50, kernel_size = (3, 3), stride = 2, device = device) # output shape: d=19, h=25, w=33
        self.pool3 = nn.MaxPool2d((3, 3), stride = 1)
        self.dropout3 = nn.Dropout2d(0.2)
        self.batchnorm3 = nn.BatchNorm2d(50, device = device)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.lstm = nn.LSTM(400, 256, device = device)
        self.linear = nn.Linear(256, 8, device = device)

        

    def forward(self, input):
        #print(f'input: {input.shape}')

        x = self.act1(self.conv1(input))
        #print(f'act1: {x.shape}')
        x = self.pool1(x)
        #print(f'pool1: {x.shape}')

        x = self.act2(self.conv2(x))
        #print(f'act2: {x.shape}')
        x = self.pool2(x)
        #print(f'pool2: {x.shape}')

        x = self.flatten(x)
        #print(f'flattened: {x.shape}')

        x, _ = self.lstm(x.view(len(input), 1, -1))
        #print(f'lstm1: {x.shape}')
        
        x = self.linear(x.view(len(input), -1))
        #print(f'lstm2: {x.shape}')

        #y_pred = nn.functional.log_softmax(x, dim=1)

        # logits = torch.tensor(x, dtype = torch.float32)

        logits = x.clone().detach().requires_grad_(True)

        return logits