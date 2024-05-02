import torch.nn as nn
import torch
import torchvision.models as models

class VGGModel(nn.Module):
    def __init__(self, device):
        super(VGGModel, self).__init__()

        self.transfer_vgg16 = models.vgg16(weights='DEFAULT').to(device) #Initialise VGG16
        for layers in self.transfer_vgg16.parameters():
            layers.requires_grad=False #Freeze layers

        dropout=0.2
        self.transfer_vgg16.classifier = nn.Sequential(
            nn.Flatten(start_dim=3, end_dim=-1),
            nn.LSTM(6912000, 256, device = device),
            nn.Linear(256, 8)
        ).to(device)



    def forward(self, input):

        x = self.transfer_vgg16.classifier(input)

        logits = x.clone().detach().requires_grad_(True)

        return logits

## Riyaad's code

# class VGGTestModel(nn.Module):
#     def __init__(self, device):
#         super(VGGTestModel, self).__init__()

#         self.network = nn.Sequential(
                    
#                     nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(64, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(64, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2, 2),
                    
#                     nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(128, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(128, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2, 2),
                    
#                     nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(256, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(256, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(256, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2, 2),
                    
#                     nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(512, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(512, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(512, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2, 2),
                    
#                     nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(512, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(512, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
#                     nn.BatchNorm2d(512, momentum = 0.9),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2, 2),
                    
#                     nn.Flatten(),
                    
#                     nn.Dropout(0.5),
#                     nn.Linear(2048, 4096),
#                     nn.ReLU(),
                    
#                     nn.Dropout(0.5),
#                     nn.Linear(4096, 4096),
#                     nn.ReLU(),
                    
#                     nn.Linear(4096, 15)

#                 ).to(device)

#         self.lstm = nn.LSTM(3200, 256, device = device)
#         self.linear = nn.Linear(256, 8, device = device)

#     def forward(self, input):
        
#         x = self.network.classifier(input)
        
#         x, _ = self.lstm(x.view(len(input), 1, -1))        
#         x = self.linear(x.view(len(input), -1))

#         logits = x.clone().detach().requires_grad_(True)

#         return logits



## ===========================================================================

# class BaselineModel(nn.Module):
#     def __init__(self, device):
#         super(BaselineModel, self).__init__()

#         # Conv Layers (videos arre 320x240)

#         # Initial kernel dims: stride=9, d=18 h=24, w=32

#         self.conv1 = nn.Conv3d(in_channels = 3, out_channels = 4, kernel_size = (18, 24, 32), stride = 9, device = device) # output shape: d=19, h=25, w=33
#         self.act1 = nn.ReLU()
#         self.pool1 = nn.MaxPool3d((2, 3, 3), stride = 1)


#         self.conv2 = nn.Conv3d(in_channels = 4, out_channels = 5, kernel_size = (4, 5, 7), stride = 2, device = device) # output shape: d=7, h=11, w=14
#         self.act2 = nn.ReLU()
#         self.pool2 = nn.MaxPool3d((1, 3, 4), stride = 1)

#         self.flatten = nn.Flatten(start_dim=2, end_dim=-1)

#         # LSTM
#         # self.lstm = nn.Sequential(
#         #     nn.LSTM(3200, 256),
#         #     nn.Linear(256, 8)
#         # )

#         self.lstm = nn.LSTM(3200, 256, device = device)
#         self.linear = nn.Linear(256, 8, device = device)

        

#     def forward(self, input):
#         #print(f'input: {input.shape}')

#         x = self.act1(self.conv1(input))
#         #print(f'act1: {x.shape}')
#         x = self.pool1(x)
#         #print(f'pool1: {x.shape}')

#         x = self.act2(self.conv2(x))
#         #print(f'act2: {x.shape}')
#         x = self.pool2(x)
#         #print(f'pool2: {x.shape}')

#         x = self.flatten(x)
#         #print(f'flattened: {x.shape}')

#         x, _ = self.lstm(x.view(len(input), 1, -1))
#         # print(f'lstm1: {x.shape}')
        
#         x = self.linear(x.view(len(input), -1))
#         #print(f'lstm2: {x.shape}')

#         #y_pred = nn.functional.log_softmax(x, dim=1)

#         # logits = torch.tensor(x, dtype = torch.float32)

#         logits = x.clone().detach().requires_grad_(True)

#         return logits
    
    