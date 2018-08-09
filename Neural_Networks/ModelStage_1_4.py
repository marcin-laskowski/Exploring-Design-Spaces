"""
MODELS OF THE NEURAL NETWORK FOR THE STAGE 1.4

input: 64x64x3 matrix with shape and 2 channels which are meshgrid with X
       and Y coordinates 
       5x6 matrix with information about nodes coordinates, fixation and forces
output: 64x64 matrix with stress distribution

"""

import torch
from torch import nn
import torch.nn.functional as F



# =============================================================================   
class ConvLinConv_0(nn.Module):
    def __init__(self):
        super(ConvLinConv_0, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,6,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU(),
            nn.Conv2d(6,12,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU(),
            nn.Conv2d(12,40,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU())                        
        self.converter = nn.Sequential(
            nn.Linear(40+30, 150),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(150, 200),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(200, 16*5*5),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16,14,2,2,0,0),
            nn.BatchNorm2d(14),
            nn.ReLU(True),
            nn.ConvTranspose2d(14,12,2,2,0,0),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12,10,2,2,0,0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,8,6,1,0,0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8,5,10,1,0,0),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.ConvTranspose2d(5,3,11,1,0,0),
            nn.ReLU())

    def forward(self, x, y):
        x = x.view(x.size(0), 3, 64, 64)
        x = self.encoder(x)
        x = x.view(x.size(0), 40)
        y = y.view(y.size(0), 30)
        x = torch.cat((x, y), 1)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 5, 5))
        x = x.view(x.size(0), 3, 64, 64)
        return x
