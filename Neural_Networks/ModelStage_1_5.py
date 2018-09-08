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
            nn.LeakyReLU(True),
            nn.Dropout(0.2),
            nn.Linear(500, 1200),
            nn.LeakyReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1200, 2000),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2000, 2000),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2000, 4000),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(4000, 3*64*64),
            nn.LeakyReLU())

    def forward(self, x):
#       x = x / torch.max(torch.abs(x))
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = x.view(x.size(0), 3, 64, 64)
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
            nn.ConvTranspose2d(16, 10, 6, stride=2),    # 10, 24, 24
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.ConvTranspose2d(10, 8, 6, stride=2),     # 6, 52, 52
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 6, 6, stride=1),      # 4, 57, 57
            nn.BatchNorm2d(6),
            nn.PReLU(),
            nn.ConvTranspose2d(6, 4, 6, stride=1),      # 3, 62, 62
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.ConvTranspose2d(4, 3, 3, stride=1),      # 1, 64, 64
            nn.PReLU())                    # Labels

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 10, 10))
        x = x.view(x.size(0), 3, 64, 64)
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
            nn.ConvTranspose2d(16, 12, 6, stride=2),    # 10, 24, 24
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 10, 6, stride=2),     # 6, 52, 52
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 8, 6, stride=1),      # 4, 57, 57
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 6, 6, stride=1),      # 3, 62, 62
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 3, stride=1),      # 1, 64, 64
            nn.PReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 10, 10))
        x = x.view(x.size(0), 3, 64, 64)
        return x



# =============================================================================
class LinConv_2(nn.Module):
    def __init__(self):
        super(LinConv_2, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 100),
            nn.PReLU(),
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
            nn.ConvTranspose2d(10, 8, 6, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 6, 6, stride=1),
            nn.BatchNorm2d(6),
            nn.PReLU(),
            nn.ConvTranspose2d(6, 4, 6, stride=1),      # 3, 62, 62
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.ConvTranspose2d(4, 3, 3, stride=1),      # 1, 64, 64
            nn.PReLU()) 

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 20, 3, 3))
        x = x.view(x.size(0), 3, 64, 64)
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
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 1200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1200, 2000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2000, 16 * 20 * 20),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16,14,2,2,0,0),      # 20, 40, 40
            nn.BatchNorm2d(14),                     
            nn.ReLU(),
            nn.ConvTranspose2d(14,12,6,1,0,0),      # 12, 45, 45
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12,10,6,1,0,0),      # 10, 50, 50
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,8,6,1,0,0),       # 8, 55, 55
            nn.ReLU(),
            nn.ConvTranspose2d(8,6,6,1,0,0),        # 6, 60, 60
            nn.PReLU(),
            nn.ConvTranspose2d(6,3,5,1,0,0),        # 3, 64, 64
            nn.PReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 20, 20))
        x = x.view(x.size(0), 3, 64, 64)
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
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 600),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(600, 50 * 4 * 4),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(50,20,2,2,0,0),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.ConvTranspose2d(20,10,2,2,0,0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,5,2,2,0,0),
            nn.BatchNorm2d(5),
            nn.PReLU(),
            nn.ConvTranspose2d(5,3,2,2,0,0),
            nn.PReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 50, 4, 4))
        x = x.view(x.size(0), 3, 64, 64)
        return x



# =============================================================================
class LinConv_5(nn.Module):
    def __init__(self):
        super(LinConv_5, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 20 * 3 * 3),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(20, 16, 3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 12, 2, stride=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 10, 6, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 8, 6, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 6, 6, stride=1),
            nn.BatchNorm2d(6),
            nn.PReLU(),
            nn.ConvTranspose2d(6, 4, 6, stride=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.ConvTranspose2d(4, 3, 3, stride=1),
            nn.PReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 20, 3, 3))
        x = x.view(x.size(0), 3, 64, 64)
        return x



# =============================================================================
class LinConv_6(nn.Module):
    def __init__(self):
        super(LinConv_6, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 20 * 3 * 3),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(20, 16, 3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 12, 2, stride=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 10, 6, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 8, 6, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 6, 6, stride=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 4, 6, stride=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.ConvTranspose2d(4, 3, 3, stride=1),
            nn.PReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 20, 3, 3))
        x = x.view(x.size(0), 3, 64, 64)
        return x



# =============================================================================
class LinConv_7(nn.Module):
    def __init__(self):
        super(LinConv_7, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 60),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(60, 180),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(180, 400),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(400, 50 * 4 * 4),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(50,20,2,2,0,0),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.ConvTranspose2d(20,10,2,2,0,0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,5,2,2,0,0),
            nn.BatchNorm2d(5),
            nn.PReLU(),
            nn.ConvTranspose2d(5,3,2,2,0,0),
            nn.PReLU())

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 50, 4, 4))
        x = x.view(x.size(0), 3, 64, 64)
        return x
    
    
    
# =============================================================================
class LinConv_8(nn.Module):
    def __init__(self):
        super(LinConv_8, self).__init__()
        self.converter= nn.Sequential(
            nn.Linear(5*6, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 40 * 10 * 10),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(40, 30, 6, stride=2),        # 24, 24
            nn.BatchNorm2d(30),
            nn.ReLU(30),
            nn.ConvTranspose2d(30, 20, 2, stride=2),        # 48, 48
            nn.BatchNorm2d(20),
            nn.ReLU(20),
            nn.ConvTranspose2d(20, 16, 6, stride=1),        # 53, 53
            nn.BatchNorm2d(16),
            nn.ReLU(16),
            nn.ConvTranspose2d(16, 10, 6, stride=1),        # 58, 58
            nn.BatchNorm2d(10),
            nn.ReLU(10),
            nn.ConvTranspose2d(10, 6, 3, stride=1),         # 60, 60
            nn.BatchNorm2d(6),
            nn.PReLU(6),
            nn.ConvTranspose2d(6, 4, 3, stride=1),          # 62, 62
            nn.BatchNorm2d(4),
            nn.PReLU(4),
            nn.ConvTranspose2d(4, 3, 3, stride=1))          # 62, 62

    def forward(self, x):
        x = x.view(x.size(0), 5*6)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 40, 10, 10))
        x = x.view(x.size(0), 3, 64, 64)
        return x
    