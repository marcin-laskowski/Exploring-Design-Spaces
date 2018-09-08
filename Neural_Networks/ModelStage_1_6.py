"""
MODELS OF THE NEURAL NETWORK FOR THE STAGE 1.3

input: 5x6 matrix with information about nodes coordinates, fixation and forces
output: 64x64 matrix with stress distribution

"""

import torch
from torch import nn
import torch.nn.functional as F



class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

# -------------------------- GENERATOR ---------------------------------------
class Generator_BA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator_BA, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)


        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.PReLU()
        )


    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)
    

# -------------------------- GENERATOR 2---------------------------------------==   
class Generator_AB(nn.Module):
    def __init__(self):
        super(Generator_AB, self).__init__()
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


# -------------------------- DISCRIMINATOR ------------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)




