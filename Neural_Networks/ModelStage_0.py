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
            nn.RReLU(0, 1))

    def forward(self, x):
        x = x.view(x.size(0), 64*64)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# =============================================================================
class ConvConv_0(nn.Module):
    def __init__(self):
        super(ConvConv_0, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=6, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(True))  # output: 1, 16, 13, 13
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, 6, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, 6, stride=2),
            nn.RReLU(0, 1))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# =============================================================================:
class ConvConv_1(nn.Module):
    def __init__(self):
        super(ConvConv_1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=6, stride=2, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(6, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1))  # output: 1, 16, 10, 10
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 10, 6, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(10, 6, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 6, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 8, stride=1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x.view(x.size(0), 16, 10, 10))
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
            nn.Sigmoid())    
        
    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x.view(x.size(0), 16 * 13 * 13))
        x = self.decoder1(x)
        x = self.decoder2(x.view(x.size(0), 16, 13, 13))
        return x


# =============================================================================
class ConvLinLinConv_1(nn.Module):
    def __init__(self):
        super(ConvLinLinConv_1, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=6, stride=2, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(6, 16, kernel_size=6, stride=2, padding=0),
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
            nn.ConvTranspose2d(16, 10, 6, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(10, 6, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 6, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 8, stride=1),
            nn.Sigmoid())   
        
    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x.view(x.size(0), 16 * 10 * 10))
        x = self.decoder1(x)
        x = self.decoder2(x.view(x.size(0), 16, 10, 10))
        return x