"""
AUTOENCODER
to obtain the latent space of the input and output data
"""

import os
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


if not os.path.exists('./model'):
    os.mkdir('./model')


# ========================= PARAMETERS ========================================
Train = False
Visualize = False

learning_rate = 0.001
num_epoch = 30
batch_size = 1
My_momentum = 0.9

lay_conf = [64*64, 256, 30]

type_of_model = 'Seq_Linear'  # Linear, Seq_linear, Convolution


# ==================== INPUT DATA =============================================
Images = Variable(torch.from_numpy(np.load('Images_DataSet.npy')))
Labels = Variable(torch.from_numpy(np.load('Images_DataSet.npy')))

Train_Images = Images[:50]
Train_Labels = Labels[:50]
Test_Images = Images[50:]
Test_Labels = Labels[50:]


# ====================== NEURAL NETWORK CONFIGURATION =========================
if type_of_model == 'Linear':
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.h1 = nn.Linear(64*64, 100)
            self.h2 = nn.Linear(100, 20)
            self.h3 = nn.Linear(20, 100)
            self.h4 = nn.Linear(100, 64*64)

        def forward(self, x):
            # x = x.view(1, 4096)
            x = x.view(-1, 64*64)
            # print(x.size())
            x = F.relu(self.h1(x))
            # print(x.size())
            x = F.relu(self.h2(x))
            x = F.relu(self.h3(x))
            x = F.relu(self.h4(x))
            return x

elif type_of_model == 'Seq_Linear':
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(lay_conf[0], lay_conf[1]),
                nn.ReLU(True),
                nn.Linear(lay_conf[1], lay_conf[2]),
                nn.ReLU(True))
            self.decoder = nn.Sequential(
                nn.Linear(lay_conf[2], lay_conf[1]),
                nn.ReLU(True),
                nn.Linear(lay_conf[1], lay_conf[0]),
                nn.Sigmoid())

        def forward(self, x):
            x = x.view(-1, lay_conf[0])
            x = self.encoder(x)
            x = self.decoder(x)
            return x

elif type_of_model == 'Convolution':
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 4),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 8, 4),
                nn.ReLU(True),
                nn.MaxPool2d(2))
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(8, 16, 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 8, 5),
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 1, 2),
                nn.Tanh())

        def forward(self, x):
            x = x.view(-1, lay_conf[0])
            x = self.encoder(x)
            x = self.decoder(x)
            return x

else:
    print("model is not specified")


# calling the model
net = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=My_momentum)

params = list(net.parameters())
loss_sum = 0


# ============================== TRAINING =====================================
if Train == True:
    # Learning process
    for epoch in range(num_epoch):

        running_loss = 0.0

        for i in range(len(Train_Images)):
            # print("= = = i: ", i, " = = =")
            inputs = Train_Images[i].float()
            labels = Train_Labels[i].float()
            inputs, labels = Variable(inputs), Variable(labels)

            net.zero_grad()

            # forward path
            out = net(inputs)
            # print('out = ', out)
            out = out.view(1, 1, 64, 64)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            loss_sum += loss.data[0]

        plt.subplot(1, 2, 1)
        plt.plot(epoch, running_loss / Images.size(0), 'b.')

        # ---------------------
        pred = net(Variable(Test_Images.float()))

        pred_im = pred.view(50, 1, 1, 64, 64)
        plt.plot(epoch, criterion(pred_im, Variable(Test_Labels.float())).data.numpy(), 'r.')

        # -------------------

        plt.hold(True)
        plt.subplot(1, 2, 2)
        plt.plot(epoch, running_loss / Images.size(0), 'b.')
        plt.hold(True)
        print("epoch : {} , loss : {}".format(epoch, running_loss / Images.size(0)))

    plt.show()

    # Save the model
    torch.save(net, "./model/image_autoencoder_model.pt")


else:
    net = torch.load('./model/image_autoencoder_model.pt')

    inputs = Test_Images.float()
    labels = Test_Labels.float()
    inputs, labels = Variable(inputs), Variable(labels)
    out = net(inputs)

    # if you want to call the encoder:
    latent_space = net.encoder(inputs.view(-1, 64*64))


# ========================= VISUALIZE =========================================
if Visualize == True:

    vis_inputs = out.view(50, 64, 64)
    vis_out = out.view(50, 64, 64)

    num_1 = 1
    num_2 = 2

    img_0 = vis_inputs[num_1, :, :]
    img_1 = vis_out[num_1, :, :]
    img_2 = vis_inputs[num_2, :, :]
    img_3 = vis_out[num_2, :, :]

    grid_z0 = (Variable(img_0).data).cpu().numpy()
    grid_z1 = (Variable(img_1).data).cpu().numpy()
    grid_z2 = (Variable(img_2).data).cpu().numpy()
    grid_z3 = (Variable(img_3).data).cpu().numpy()

    plt.subplot(221)
    plt.imshow(grid_z0.T, extent=(0, 64, 0, 64), origin='1')
    plt.title('1')
    plt.subplot(222)
    plt.imshow(grid_z1.T, extent=(0, 64, 0, 64), origin='2')
    plt.title('2')
    plt.subplot(223)
    plt.imshow(grid_z2.T, extent=(0, 64, 0, 64), origin='3')
    plt.title('3')
    plt.subplot(224)
    plt.imshow(grid_z3.T, extent=(0, 64, 0, 64), origin='4')
    plt.title('4')
    plt.show()
