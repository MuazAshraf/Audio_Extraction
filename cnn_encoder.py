import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    """
    CNN Encoder to extract features from Mel Spectrograms.
    Input: (Batch, Channels, Freq, Time) -> e.g., (B, 1, 128, T)
    Output: (Batch, Time', Features)
    """
    def __init__(self, in_channels=1, base_channels=32):
        super(EncoderCNN, self).__init__()
        
        # Convolutional layers
        # Stride 2 in time reduces the time dimension size
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=(1, 1), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces Freq and Time by 2
        
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=(1, 1), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces Freq and Time by 2 again (Total /4)
        
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=(1, 1), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces Freq and Time by 2 again (Total /8)
        
        self.flat_feature_dim = 16 * (base_channels * 4)
        
        # Linear projection to RNN size or just keep as is
        self.fc = nn.Linear(self.flat_feature_dim, 512) 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x))) 
            
        x = x.permute(0, 3, 1, 2) 
        
        batch, time, ch, freq = x.size()
        x = x.reshape(batch, time, ch * freq)
        # Shape: (Batch, Time', Flattened_Features)
        
        x = self.dropout(self.fc(x))
        return x
