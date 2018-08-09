"""
MODELS OF THE NEURAL NETWORK FOR THE STAGE 1.3

input: 5x6 matrix with information about nodes coordinates, fixation and forces
output: 64x64 matrix with stress distribution

"""

import torch
from torch import nn
import torch.nn.functional as F



# =============================================================================
class Lin_0(nn.Module):
    def __init__(self):
        super(Lin_0, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 500),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(500, 1200),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1200, 2000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2000, 64*64),
            nn.ReLU())

    def forward(self, x):
#                    x = x / torch.max(torch.abs(x))
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = x.view(x.size(0), 1, 64, 64)
        return x


# =============================================================================
class LinConv_0(nn.Module):
    def __init__(self):
        super(LinConv_0, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 16*10*10),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16, 10, 6, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.ConvTranspose2d(10, 6, 6, stride=2),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, 6, stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 8, stride=1),
            nn.ReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 10, 10))
        x = x.view(x.size(0), 1, 64, 64)
        return x



# =============================================================================
class LinConv_1(nn.Module):
    def __init__(self):
        super(LinConv_1, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 500),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(500, 1200),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1200, 16 * 10 * 10),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16, 10, 6, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.ConvTranspose2d(10, 6, 6, stride=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 6, stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 8, stride=1),
            nn.ReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 10, 10))
        x = x.view(x.size(0), 1, 64, 64)
        return x



# =============================================================================
class LinConv_2(nn.Module):
    def __init__(self):
        super(LinConv_2, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 200),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(200, 20 * 3 * 3),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(20, 16, 3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 12, 2, stride=2),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 10, 6, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.ConvTranspose2d(10, 6, 6, stride=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 6, stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 8, stride=1),
            nn.ReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 20, 3, 3))
        x = x.view(x.size(0), 1, 64, 64)
        return x


# =============================================================================
class LinConv_3(nn.Module):
    def __init__(self):
        super(LinConv_3, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 500),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(500, 1200),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1200, 50 * 4 * 4),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(50,20,2,2,0,0),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.ConvTranspose2d(20,10,2,2,0,0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,5,2,2,0,0),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.ConvTranspose2d(5,1,2,2,0,0),
            nn.ReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 50, 4, 4))
        x = x.view(x.size(0), 1, 64, 64)
        return x
    


# =============================================================================
class LinConv_4(nn.Module):
    def __init__(self):
        super(LinConv_4, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 400),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(400, 600),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(600, 50 * 4 * 4),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(50,20,2,2,0,0),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.ConvTranspose2d(20,10,2,2,0,0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,5,2,2,0,0),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.ConvTranspose2d(5,1,2,2,0,0),
            nn.ReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 50, 4, 4))
        x = x.view(x.size(0), 1, 64, 64)
        return x



# =============================================================================
class LinConv_5(nn.Module):
    def __init__(self):
        super(LinConv_5, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 200),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(200, 20 * 3 * 3),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(20, 16, 3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 12, 2, stride=2),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 10, 6, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.ConvTranspose2d(10, 6, 6, stride=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 6, stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 8, stride=1),
            nn.ReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 20, 3, 3))
        x = x.view(x.size(0), 1, 64, 64)
        return x



# =============================================================================
class LinConv_6(nn.Module):
    def __init__(self):
        super(LinConv_6, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 60),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(60, 180),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(180, 400),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(400, 50 * 4 * 4),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(50,20,2,2,0,0),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.ConvTranspose2d(20,10,2,2,0,0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,5,2,2,0,0),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.ConvTranspose2d(5,1,2,2,0,0),
            nn.ReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 50, 4, 4))
        x = x.view(x.size(0), 1, 64, 64)
        return x
    