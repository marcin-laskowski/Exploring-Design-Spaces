"""
MODELS OF THE NEURAL NETWORK FOR THE STAGE 1.2

input: 64x64 matrix with shape and 5x6 matrix with information about nodes
       coordinates, fixation and forces
output: 64x64 matrix with stress distribution

"""

import torch
from torch import nn
import torch.nn.functional as F



# =============================================================================   
class LinLinConv_0(nn.Module):
    def __init__(self):
        super(LinLinConv_0, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64*64, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(500, 100),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(100, 40),
            nn.ReLU())
        self.converter = nn.Sequential(
            nn.Linear(60, 200),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(200, 500),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(500, 16*10*10),
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

    def forward(self, x, y):
        x = x.view(x.size(0), 64*64)
        x = self.encoder(x)
        x = x.view(x.size(0), 40)
        y = y.view(y.size(0), 20)
        x = torch.cat((x, y), 1)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 10, 10))
        x = x.view(x.size(0), 1, 64, 64)
        return x



# =============================================================================
class LinLinConv_1(nn.Module):
    def __init__(self):
        super(LinLinConv_1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64*64, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.Linear(500, 100),
            nn.ReLU(True),
            nn.Linear(100, 40),
            nn.ReLU())
        self.converter = nn.Sequential(
            nn.Linear(60, 100),
            nn.ReLU(True),
            nn.Linear(100, 500),
            nn.ReLU(True),
            nn.Linear(500, 16*10*10),
            nn.ReLU(True),
            nn.Dropout())
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

    def forward(self, x, y):
        x = x.view(x.size(0), 64*64)
        x = self.encoder(x)
        x = x.view(x.size(0), 40)
        y = y.view(y.size(0), 20)
        x = torch.cat((x, y), 1)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 10, 10))
        x = x.view(x.size(0), 1, 64, 64)
        return x



# =============================================================================
class ConvLinConv_0(nn.Module):
    def __init__(self):
        super(ConvLinConv_0, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,6,3,1,1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(6,12,3,1,1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(12,20,3,1,1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(20,40,3,1,1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2,2))                        
        self.converter = nn.Sequential(
            nn.Linear(40+20, 200),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(200, 500),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(500, 300),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(300, 200),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(200,50,4,4,0,0),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
            nn.ConvTranspose2d(50,10,4,4,0,0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,5,2,2,0,0),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.ConvTranspose2d(5,2,2,2,0,0),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.ConvTranspose2d(2,1,1,1,0,0),
            nn.ReLU())

    def forward(self, x, y):
        x = x.view(x.size(0), 1, 64, 64)
        x = self.encoder(x)
        x = x.view(x.size(0), 40)
        y = y.view(y.size(0), 20)
        x = torch.cat((x, y), 1)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 200, 1, 1))
        x = x.view(x.size(0), 1, 64, 64)
        return x



# =============================================================================
class ConvLinConv_1(nn.Module):
    def __init__(self):
        super(ConvLinConv_1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,6,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU(),
            nn.Conv2d(6,12,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU(),
            nn.Conv2d(12,40,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU())                        
        self.converter = nn.Sequential(
            nn.Linear(40+20, 200),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(200, 400),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(400, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, 16*5*5),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16,12,2,2,0,0),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.ConvTranspose2d(12,10,2,2,0,0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,8,2,2,0,0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8,6,6,1,0,0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6,4,10,1,0,0),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4,1,11,1,0,0),
            nn.ReLU())

    def forward(self, x, y):
        x = x.view(x.size(0), 1, 64, 64)
        x = self.encoder(x)
        x = x.view(x.size(0), 40)
        y = y.view(y.size(0), 20)
        x = torch.cat((x, y), 1)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 5, 5))
        x = x.view(x.size(0), 1, 64, 64)
        return x



# =============================================================================
class ConvLinConv_2(nn.Module):
    def __init__(self):
        super(ConvLinConv_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,6,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU(),
            nn.Conv2d(6,12,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU(),
            nn.Conv2d(12,40,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU())                        
        self.converter = nn.Sequential(
            nn.Linear(40+20, 200),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(200, 400),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(400, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, 16*10*10),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16,12,2,2,0,0),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12,10,2,2,0,0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,7,6,1,0,0),
            nn.BatchNorm2d(7),
            nn.ReLU(),
            nn.ConvTranspose2d(7,5,10,1,0,0),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.ConvTranspose2d(5,1,11,1,0,0),
            nn.ReLU())

    def forward(self, x, y):
        x = x.view(x.size(0), 1, 64, 64)
        x = self.encoder(x)
        x = x.view(x.size(0), 40)
        y = y.view(y.size(0), 20)
        x = torch.cat((x, y), 1)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 10, 10))
        x = x.view(x.size(0), 1, 64, 64)
        return x


# =============================================================================
class ConvLinConv_3(nn.Module):
    def __init__(self):
        super(ConvLinConv_3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,6,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU(),
            nn.Conv2d(6,12,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU(),
            nn.Conv2d(12,40,3,1,1),
            nn.MaxPool2d(4,4),
            nn.ReLU())                        
        self.converter = nn.Sequential(
            nn.Linear(40+20, 150),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(150, 200),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(200, 16*5*5),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16,12,2,2,0,0),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.ConvTranspose2d(12,10,2,2,0,0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,8,2,2,0,0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8,6,6,1,0,0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6,4,10,1,0,0),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4,1,11,1,0,0),
            nn.ReLU())

    def forward(self, x, y):
        x = x.view(x.size(0), 1, 64, 64)
        x = self.encoder(x)
        x = x.view(x.size(0), 40)
        y = y.view(y.size(0), 20)
        x = torch.cat((x, y), 1)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 5, 5))
        x = x.view(x.size(0), 1, 64, 64)
        return x



# =============================================================================
class ConvLinConv_4(nn.Module):
    def __init__(self):
        super(ConvLinConv_4, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,6,3,1,1),               # 6, 64, 64
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                  # 6, 32, 32
            nn.Conv2d(6,10,3,1,1),              # 10, 32, 32
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                  # 10, 16, 16
            nn.Conv2d(10,14,3,1,1),             # 14, 16, 16
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                  # 12, 8, 8
            nn.Conv2d(14,20,3,1,1),             # 20, 8, 8
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                  # 20, 4, 4
            nn.Conv2d(20,25,3,1,1),             # 25, 4, 4
            nn.BatchNorm2d(25),
            nn.ReLU())
        self.converter = nn.Sequential(
            nn.Linear((25*4*4)+20, 500),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(500, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, 2000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2000, 2000),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(2000, 1000),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1000, 16*5*5),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16,12,2,2,0,0),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.ConvTranspose2d(12,10,2,2,0,0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10,8,2,2,0,0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8,6,6,1,0,0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6,4,10,1,0,0),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4,1,11,1,0,0),
            nn.ReLU())

    def forward(self, x, y):
        x = x.view(x.size(0), 1, 64, 64)
        x = self.encoder(x)
        x = x.view(x.size(0), 25*4*4)
        y = y.view(y.size(0), 20)
        x = torch.cat((x, y), 1)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 5, 5))
        x = x.view(x.size(0), 1, 64, 64)
        return x