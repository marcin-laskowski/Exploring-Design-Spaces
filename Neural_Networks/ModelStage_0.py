"""
MODELS OF THE NEURAL NETWORK FOR THE STAGE 0

The goal of this stage is to investigate different types of autoencoders

input = ouput: 64x64 matrix with stress distribution or 
               64x64 matrix with shape of the pentagon

"""

import torch
from torch import nn
import torch.nn.functional as F



# =============================================================================
class FunLin_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(64*64, 300)
        self.h2 = nn.Linear(300, 30)
        self.h3 = nn.Linear(30, 300)
        self.h4 = nn.Linear(300, 64*64)

    def forward(self, x):
        x = x.view(x.size(0), 64*64)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = F.relu(self.h4(x))
        return x


# =============================================================================
class LinLin_0(nn.Module):
    def __init__(self):
        super(LinLin_0, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64*64, 200),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(200, 40),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(40,200),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(200, 64*64),
#            nn.ReLU())                      # Labels
            nn.Sigmoid())                   # Inputs

    def forward(self, x):
        x = x.view(x.size(0), 64*64)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 64, 64)
        return x
    

# =============================================================================
class LinLin_1(nn.Module):
    def __init__(self):
        super(LinLin_1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64*64, 500),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 40),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(40,100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100,500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 64*64),
#            nn.ReLU())                      # Labels
            nn.Sigmoid())                   # Inputs

    def forward(self, x):
        x = x.view(x.size(0), 64*64)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 64, 64)
        return x


# =============================================================================
class ConvConv_0(nn.Module):
    def __init__(self):
        super(ConvConv_0, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,3,3,1,1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(3,6,3,1,1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,8,3,1,1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(8,10,3,1,1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(10,12,3,1,1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2,2))                      # 12, 2, 2  
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(12,10,4,2,1,0),      # 10, 4, 4
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,8,4,2,1,0),       # 8, 8, 8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8,6,4,2,1,0),        # 6, 16, 16
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6,3,4,2,1,0),        # 3, 32, 32
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(3,1,4,2,1,0),        # 1, 64, 64
#            nn.ReLU())                      # Labels
            nn.Sigmoid())                   # Inputs

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 64, 64)
        return x


# =============================================================================:
class ConvConv_1(nn.Module):
    def __init__(self):
        super(ConvConv_1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 3, 1, 0),           # 6, 62, 62
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                 # 6, 31, 31
            nn.Conv2d(6, 10, 4, 1, 0),          # 10, 28, 28
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                 # 10, 14, 14
            nn.Conv2d(10, 12, 5, 1, 0),         # 12, 10, 10
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2))                 # 12, 5, 5
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(12, 11, 4, 2, 1),               # 12, 10, 10 
            nn.BatchNorm2d(11),
            nn.ReLU(True),
            nn.ConvTranspose2d(11, 10, 6, 2, 1),               # 11, 22, 22
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 9, 6, 2, 1),               # 9, 46, 46
            nn.BatchNorm2d(9),
            nn.ReLU(),
            nn.ConvTranspose2d(9, 8, 8, 1, 1),                # 8, 51, 51
            nn.BatchNorm2d(8),
            nn.ReLU(),            
            nn.ConvTranspose2d(8, 6, 8, 1, 1),                 # 6, 56, 56
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 4, 8, 1, 1),                 # 4, 61, 61
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 6, 1, 1),                 # 1, 64, 64
#            nn.ReLU())                      # Labels
            nn.Sigmoid())                   # Inputs

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 64, 64)
        return x


# =============================================================================
class ConvLinLinConv_0(nn.Module):
    def __init__(self):
        super(ConvLinLinConv_0, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=6, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(True))  # output: 1, 16, 13, 13
        self.encoder2 = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 40),
            nn.ReLU())
        self.decoder1 = nn.Sequential(
            nn.Linear(40, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 16 * 13 * 13),
            nn.ReLU())
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 6, 6, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, 6, stride=2),
#            nn.ReLU())                      # Labels
            nn.Sigmoid())                   # Inputs
        
    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x.view(x.size(0), 16 * 13 * 13))
        x = self.decoder1(x)
        x = self.decoder2(x.view(x.size(0), 16, 13, 13))
        x = x.view(x.size(0), 1, 64, 64)
        return x


# =============================================================================
class ConvLinLinConv_1(nn.Module):
    def __init__(self):
        super(ConvLinLinConv_1, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=6, stride=2, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(6, 16, kernel_size=6, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1))  # output: 1, 16, 10, 10
        self.encoder2 = nn.Sequential(
            nn.Linear(16 * 10 * 10, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 40),
            nn.ReLU())
        self.decoder1 = nn.Sequential(
            nn.Linear(40, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 16 * 10 * 10),
            nn.ReLU())
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 10, 6, stride=2),    # 10, 24, 24
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.ConvTranspose2d(10, 6, 6, stride=2),     # 6, 52, 52
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 4, 6, stride=1),      # 4, 57, 57
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, 6, stride=1),      # 3, 62, 62
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 3, stride=1),      # 1, 64, 64
#            nn.ReLU())                      # Labels
            nn.Sigmoid())                   # Inputs 
        
    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x.view(x.size(0), 16 * 10 * 10))
        x = self.decoder1(x)
        x = self.decoder2(x.view(x.size(0), 16, 10, 10))
        x = x.view(x.size(0), 1, 64, 64)
        return x