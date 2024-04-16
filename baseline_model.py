import torch.nn as nn

class baseline_model(nn.Module):
    def __init__(self):
        super(baseline_model, self).__init__()

        # Conv Layers (videos arre 320x240)

        self.conv1 = nn.Conv2d()
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d()

        self.conv1 = nn.Conv2d()
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d()

        self.flatten = nn.Flatten()

        # LSTM
        self.model = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim), 
            nn.LSTM(embedding_dim, hidden_dim), 
            nn.Linear(hidden_dim, labels_size)
            )

        

    def forward(self, input):
        
        y_pred = 0
        return y_pred