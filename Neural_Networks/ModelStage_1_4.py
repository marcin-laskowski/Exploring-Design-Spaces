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
            nn.Linear(40+30, 150),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(150, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 16*5*5),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16,14,2,2,0,0),
            nn.BatchNorm2d(14),
            nn.ReLU(),
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
            nn.ConvTranspose2d(5,3,11,1,0,0))
#            nn.PReLU())

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
    

# =============================================================================   
class ConvLinConv_1(nn.Module):
    def __init__(self):
        super(ConvLinConv_1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,6,3,1,1),
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
            nn.Linear(40+30, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 16*5*5),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16,14,2,2,0,0),
            nn.BatchNorm2d(14),
            nn.ReLU(),
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
            nn.ConvTranspose2d(5,3,11,1,0,0))
#            nn.PReLU())

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



# =============================================================================   
class ConvLinConv_2(nn.Module):
    def __init__(self):
        super(ConvLinConv_2, self).__init__()
        self.encoder = nn.Sequential(       # 3, 64, 64
            nn.Conv2d(3,6,3,1,1),           # 6, 64, 64   
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2,2),              # 6, 32, 32  
            nn.Conv2d(6,10,3,1,1),          # 10, 32, 32  
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2,2),              # 10, 16, 16 
            nn.Conv2d(10,14,3,1,1),         # 14, 16, 16
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.MaxPool2d(2,2),              # 14, 8, 8
            nn.Conv2d(14,16,3,1,1),         # 16, 8, 8
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),              # 16, 4, 4
            nn.Conv2d(16,20,3,1,1),         # 20, 4, 4
            nn.ReLU(),
            nn.MaxPool2d(2,2))              # 20, 2, 2                  
        self.converter = nn.Sequential(
            nn.Linear(80+30, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 16*5*5))
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16,14,2,2,0,0),
            nn.BatchNorm2d(14),
            nn.ReLU(),
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
            nn.ConvTranspose2d(5,3,11,1,0,0))
#            nn.PReLU())

    def forward(self, x, y):
        x = x.view(x.size(0), 3, 64, 64)
        x = self.encoder(x)
        x = x.view(x.size(0), 80)
        y = y.view(y.size(0), 30)
        x = torch.cat((x, y), 1)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 5, 5))
        x = x.view(x.size(0), 3, 64, 64)
        return x
    
    
# =============================================================================   
class ConvLinConv_3(nn.Module):
    def __init__(self):
        super(ConvLinConv_3, self).__init__()
        self.encoder = nn.Sequential(       # 3, 64, 64
            nn.Conv2d(3,6,3,1,1),           # 6, 64, 64   
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2,2),              # 6, 32, 32  
            nn.Conv2d(6,10,3,1,1),          # 10, 32, 32  
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2,2),              # 10, 16, 16 
            nn.Conv2d(10,14,3,1,1),         # 14, 16, 16
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.MaxPool2d(2,2),              # 14, 8, 8
            nn.Conv2d(14,16,3,1,1),         # 16, 8, 8
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),              # 16, 4, 4
            nn.Conv2d(16,20,3,1,1),         # 20, 4, 4
            nn.ReLU(),
            nn.MaxPool2d(2,2))              # 20, 2, 2  
        self.encoder_lin = nn.Sequential(
            nn.Linear(80, 60))                   
        self.converter = nn.Sequential(
            nn.Linear(60+30, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 16*5*5))
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16,14,2,2,0,0),
            nn.BatchNorm2d(14),
            nn.ReLU(),
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
            nn.ConvTranspose2d(5,3,11,1,0,0))
#            nn.PReLU())

    def forward(self, x, y):
        x = x.view(x.size(0), 3, 64, 64)
        x = self.encoder(x)
        x = self.encoder_lin(x.view(x.size(0), 80))
        x = x.view(x.size(0), 60)
        y = y.view(y.size(0), 30)
        x = torch.cat((x, y), 1)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 5, 5))
        x = x.view(x.size(0), 3, 64, 64)
        return x
    
    
# =============================================================================   
#class ConvLinConv_4(nn.Module):
#    def __init__(self):
#        super(ConvLinConv_4, self).__init__()
#        self.encoder = nn.Sequential(       # 3, 64, 64
#            nn.Conv2d(3,6,3,1,1),           # 6, 64, 64   
#            nn.BatchNorm2d(6),
#            nn.MaxPool2d(2,2, return_indices=True),              # 6, 32, 32  
#            nn.ReLU(),
#            nn.Conv2d(6,10,3,1,1),          # 10, 32, 32  
#            nn.BatchNorm2d(10),
#            nn.MaxPool2d(2,2, return_indices=True),              # 10, 16, 16 
#            nn.ReLU(),
#            nn.Conv2d(10,14,3,1,1),         # 14, 16, 16
#            nn.BatchNorm2d(14),
#            nn.MaxPool2d(2,2, return_indices=True),              # 14, 8, 8
#            nn.ReLU(),
#            nn.Conv2d(14,16,3,1,1),         # 16, 8, 8
#            nn.BatchNorm2d(16),
#            nn.MaxPool2d(2,2, return_indices=True),              # 16, 4, 4
#            nn.ReLU(),
#            nn.Conv2d(16,20,3,1,1),         # 20, 4, 4
#            nn.MaxPool2d(2,2, return_indices=True),              # 20, 2, 2   
#            nn.ReLU())  
#        self.encoder_lin = nn.Sequential(
#            nn.Linear(80, 60),
#            nn.ReLU())                
#        self.converter = nn.Sequential(
#            nn.Linear(60+30, 200),
#            nn.ReLU(True),
#            nn.Dropout(0.2),
#            nn.Linear(200, 400),
#            nn.ReLU(True),
#            nn.Dropout(0.5),
#            nn.Linear(400, 400),
#            nn.ReLU(True),
#            nn.Dropout(0.5),
#            nn.Linear(400, 20*2*2),
#            nn.ReLU())
#        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
#            nn.ConvTranspose2d(20,16,3,1,1,0),
#            nn.BatchNorm2d(14),
#            nn.MaxUnpool2d(2, 2),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(16,14,3,1,1),
#            nn.BatchNorm2d(14),
#            nn.MaxUnpool2d(2, 2),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(14,10,3,1,1),
#            nn.BatchNorm2d(10),
#            nn.MaxUnpool2d(2, 2),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(10,6,3,1,1),
#            nn.BatchNorm2d(6),
#            nn.MaxUnpool2d(2, 2),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(6,3,3,1,1),
#            nn.BatchNorm2d(3),
#            nn.MaxUnpool2d(2, 2),
#            nn.ReLU(True))
#
#    def forward(self, x, y):
#        x = x.view(x.size(0), 3, 64, 64)
#        x = self.encoder(x)
#        x = self.encoder_lin(x.view(x.size(0), 80))
#        x = x.view(x.size(0), 60)
#        y = y.view(y.size(0), 30)
#        x = torch.cat((x, y), 1)
#        x = self.converter(x)
#        x = self.decoder(x.view(x.size(0), 20, 2, 2))
#        x = x.view(x.size(0), 3, 64, 64)
#        return x
#    
#
## =============================================================================   
#class ConvLinConv_5(nn.Module):
#    def __init__(self):
#        super(ConvLinConv_5, self).__init__()
#        self.encoder = nn.Sequential(       # 3, 64, 64
#            nn.Conv2d(3,6,3,1,1),           # 6, 64, 64   
#            nn.BatchNorm2d(6),
#            nn.MaxPool2d(2,2, return_indices=True),              # 6, 32, 32  
#            nn.ReLU(),
#            nn.Conv2d(6,10,3,1,1),          # 10, 32, 32  
#            nn.BatchNorm2d(10),
#            nn.MaxPool2d(2,2, return_indices=True),              # 10, 16, 16 
#            nn.ReLU(),
#            nn.Conv2d(10,14,3,1,1),         # 14, 16, 16
#            nn.BatchNorm2d(14),
#            nn.MaxPool2d(2,2, return_indices=True),              # 14, 8, 8
#            nn.ReLU(),
#            nn.Conv2d(14,16,3,1,1),         # 16, 8, 8
#            nn.BatchNorm2d(16),
#            nn.MaxPool2d(2,2, return_indices=True),              # 16, 4, 4
#            nn.ReLU(),
#            nn.Conv2d(16,20,3,1,1),         # 20, 4, 4
#            nn.MaxPool2d(2,2, return_indices=True),              # 20, 2, 2   
#            nn.ReLU())  
#        self.encoder_lin = nn.Sequential(
#            nn.Linear(80, 60),
#            nn.ReLU())                
#        self.converter = nn.Sequential(
#            nn.Linear(60+30, 200),
#            nn.ReLU(True),
#            nn.Dropout(0.2),
#            nn.Linear(200, 400),
#            nn.ReLU(True),
#            nn.Dropout(0.5),
#            nn.Linear(400, 400),
#            nn.ReLU(True),
#            nn.Dropout(0.5),
#            nn.Linear(400, 16*4*4),
#            nn.ReLU())
#        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
#            nn.ConvTranspose2d(16,14,3,1,1),
#            nn.BatchNorm2d(14),
#            nn.MaxUnpool2d(2, 2),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(14,10,3,1,1),
#            nn.BatchNorm2d(10),
#            nn.MaxUnpool2d(2, 2),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(10,6,3,1,1),
#            nn.BatchNorm2d(6),
#            nn.MaxUnpool2d(2, 2),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(6,3,3,1,1),
#            nn.BatchNorm2d(3),
#            nn.MaxUnpool2d(2, 2),
#            nn.ReLU(True))
#
#    def forward(self, x, y):
#        x = x.view(x.size(0), 3, 64, 64)
#        x = self.encoder(x)
#        x = self.encoder_lin(x.view(x.size(0), 80))
#        x = x.view(x.size(0), 60)
#        y = y.view(y.size(0), 30)
#        x = torch.cat((x, y), 1)
#        x = self.converter(x)
#        x = self.decoder(x.view(x.size(0), 16, 4, 4))
#        x = x.view(x.size(0), 3, 64, 64)
#        return x
        

# =============================================================================
class ConvLinConv_6(nn.Module):
    def __init__(self):
        super(ConvLinConv_6, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,6,3,1,1),               # 6, 64, 64
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
            nn.Linear((25*4*4)+30, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 16*5*5),
            nn.ReLU())
        self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
            nn.ConvTranspose2d(16,12,2,2,0,0),
            nn.BatchNorm2d(12),
            nn.ReLU(),
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
            nn.PReLU(),
            nn.ConvTranspose2d(4,3,11,1,0,0))

    def forward(self, x, y):
        x = x.view(x.size(0), 3, 64, 64)
        x = self.encoder(x)
        x = x.view(x.size(0), 25*4*4)
        y = y.view(y.size(0), 30)
        x = torch.cat((x, y), 1)
        x = self.converter(x)
        x = self.decoder(x.view(x.size(0), 16, 5, 5))
        x = x.view(x.size(0), 3, 64, 64)
        return x