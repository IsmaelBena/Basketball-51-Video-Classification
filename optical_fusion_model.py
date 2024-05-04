import torch.nn as nn
import torch

class OpticalFusionModel(nn.Module):
    def __init__(self, device):
        super(OpticalFusionModel, self).__init__()

        # Conv Layers (videos arre 320x240)

        # Initial kernel dims: stride=9, d=18 h=24, w=32

        self.iconv1 = nn.Conv3d(in_channels = 3, out_channels = 24, kernel_size = (5, 3, 3), stride = 3, device = device) # output shape: d=29, h=21, w=21
        self.ipool1 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.idropout1 = nn.Dropout3d(0.2)
        self.ibatchnorm1 = nn.BatchNorm3d(24, device = device)

        self.iconv2 = nn.Conv3d(in_channels = 24, out_channels = 48, kernel_size = (2, 2, 3), stride = 2, device = device) # output shape: d=14, h=10, w=10
        self.ipool2 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.idropout2 = nn.Dropout3d(0.2)
        self.ibatchnorm2 = nn.BatchNorm3d(48, device = device)

        self.iconv3 = nn.Conv3d(in_channels = 48, out_channels = 64, kernel_size = (2, 2, 3), stride = 2, device = device) # output shape: d=19, h=25, w=33
        self.ipool3 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.idropout3 = nn.Dropout3d(0.2)
        self.ibatchnorm3 = nn.BatchNorm3d(64, device = device)

        self.oconv1 = nn.Conv3d(in_channels = 3, out_channels = 24, kernel_size = (5, 3, 3), stride = 3, device = device) # output shape: d=29, h=21, w=21
        self.opool1 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.odropout1 = nn.Dropout3d(0.2)
        self.obatchnorm1 = nn.BatchNorm3d(24, device = device)

        self.oconv2 = nn.Conv3d(in_channels = 24, out_channels = 48, kernel_size = (2, 2, 3), stride = 2, device = device) # output shape: d=14, h=10, w=10
        self.opool2 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.odropout2 = nn.Dropout3d(0.2)
        self.obatchnorm2 = nn.BatchNorm3d(48, device = device)

        self.oconv3 = nn.Conv3d(in_channels = 48, out_channels = 64, kernel_size = (2, 2, 3), stride = 2, device = device) # output shape: d=19, h=25, w=33
        self.opool3 = nn.MaxPool3d((2, 3, 3), stride = 1)
        self.odropout3 = nn.Dropout3d(0.2)
        self.obatchnorm3 = nn.BatchNorm3d(64, device = device)

        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)

        # LSTM
        # self.lstm = nn.Sequential(
        #     nn.LSTM(3200, 256),
        #     nn.Linear(256, 8)
        # )

        self.ilstm = nn.LSTM(320, 256, device = device)
        self.olstm = nn.LSTM(320, 256, device = device)

        self.linear = nn.Linear(512, 8, device = device)

    def forward(self, i_input, o_input):
        # print(f'input: {i_input.shape}')
        # print(f'input: {o_input.shape}')

        ix = self.iconv1(i_input)
        ix = self.ipool1(ix)
        ix = self.idropout1(ix)
        ix = self.ibatchnorm1(ix)
        
        ix = self.iconv2(ix)
        ix = self.ipool2(ix)
        ix = self.idropout2(ix)
        ix = self.ibatchnorm2(ix)

        ix = self.iconv3(ix)
        ix = self.ipool3(ix)
        ix = self.idropout3(ix)
        ix = self.ibatchnorm3(ix)

        ox = self.oconv1(o_input)
        ox = self.opool1(ox)
        ox = self.odropout1(ox)
        ox = self.obatchnorm1(ox)
        
        ox = self.oconv2(ox)
        ox = self.opool2(ox)
        ox = self.odropout2(ox)
        ox = self.obatchnorm2(ox)

        ox = self.oconv3(ox)
        ox = self.opool3(ox)
        ox = self.odropout3(ox)
        ox = self.obatchnorm3(ox)

        ix = self.flatten(ix)
        ox = self.flatten(ox)
        #print(f'flattened: {x.shape}')

        ix, _ = self.ilstm(ix.view(len(i_input), 1, -1))
        ox, _ = self.olstm(ox.view(len(o_input), 1, -1))
        # print(f'lstm1: {ix.shape}')
        # print(f'lstm1: {ox.shape}')
        ix = torch.reshape(ix, (ix.shape[0], ix.shape[-1]))
        ox = torch.reshape(ox, (ox.shape[0], ox.shape[-1]))
        # print(f'rei: {ix.shape}')
        # print(f'reo: {ox.shape}')

        x = torch.cat((ix, ox), dim=1)
        # print(x.shape)
        
        x = self.linear(x)
        #print(f'lstm2: {x.shape}')

        #y_pred = nn.functional.log_softmax(x, dim=1)

        # logits = torch.tensor(x, dtype = torch.float32)

        logits = x.clone().detach().requires_grad_(True)

        return logits
    
    