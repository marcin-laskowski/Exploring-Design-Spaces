import os
import torch
from torch import nn, optim
import numpy as np
import time
from math import sqrt
from IPython import get_ipython

import VisualizationFunction as VF

import warnings
warnings.filterwarnings("ignore")

start_time = 0
start_time = time.time()


# ========================= PARAMETERS ========================================
stage = 'Neural_Net'

Train = 1

device_type = 'cuda'  # cpu or cuda
learning_rate = 0.001
num_epoch = 10
mini_batch_size = 1000
momentum = 0.9
img_size = 64
num_channels = 1

split = 6/10

validataion_model = '.pt'


# ====================== INITIALIZE DEVICE ====================================
device = torch.device(device_type)


# ==================== INPUT DATA =============================================
Inputs = np.load('Images_DataSet.npy')
Labels = np.load('Images_DataSet.npy')


Inputs = Inputs[:5000]
Labels = Labels[:5000]
train_inputs, train_labels, test_inputs, test_labels = VF.load_data(Inputs, Labels, split, mini_batch_size, device, img_size)



# ====================== NEURAL NETWORK CONFIGURATION =========================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
            nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x.view(x.size(0), 16, 10, 10))
        return x    



# calling the model
net = Net().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

params = list(net.parameters())
loss_sum = 0
best_test_value = 0

train_test_loss = np.zeros((num_epoch, 2))
all_store_data = np.zeros((num_epoch, 5))



# ============================== TRAINING =====================================
if Train == True:
    # Learning process
    for epoch in range(num_epoch):
        
        running_loss = 0.0

        for i in range(mini_batch_size):

            # -------------------- TRAIN DATA ---------------------------------            
            train_input = train_inputs[:, i, :, :, :]  # train_input.size --> (:, 1, 64, 64)
            train_label = train_labels[:, i, :, :, :]  # train_labels.size --> (:, 1, 64, 64)
            
            # forward path
            train_out = net(train_input)
            train_out = train_out.view(train_out.size(0), num_channels, img_size, img_size)
            
            loss = criterion(train_out, train_label)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            loss_sum += loss.data[0]
            
            net.zero_grad()

            

            
        end_time = time.time()
        
        epoch_data, train_outputs, test_outputs = VF.epoch_progress(epoch, num_epoch, mini_batch_size, net, train_inputs, train_labels,
                                                                    test_inputs, test_labels, criterion, start_time,
                                                                    end_time, img_size, device)
        
        all_store_data[epoch, :] = epoch_data
        
    final_time = time.time()
    
    # plot results    
    VF.result_plot(stage, all_store_data, criterion, True, saveSVG=True)
    VF.plot_output(stage, num_epoch, test_labels, test_outputs, img_size, True, saveSVG=True)
        
    # save model and state_dict
    VF.save_model(net, 'Autoencoder_' + stage)
    
    # obtain image difference
    img_diff = VF.image_difference(test_labels, test_outputs)
    
    time.sleep(5)
    
    # save all parameters
    model_specification = VF.net_specification(stage, train_input.size(0), 
                                               str(train_inputs.size(3)) +' x ' + str(train_inputs.size(4)),
                                               str(test_inputs.size(3)) +' x ' + str(test_inputs.size(4)),
                                               num_epoch, mini_batch_size, learning_rate, momentum, criterion,
                                               optimizer, str(train_inputs.size(0)*train_inputs.size(1)) + ' / ' + str(test_inputs.size(0)*test_inputs.size(1)),
                                               device_type, np.min(all_store_data[:,1]), np.max(all_store_data[:,1]),
                                               np.min(all_store_data[:,2]), np.max(all_store_data[:,2]),
                                               sqrt(all_store_data[num_epoch-1, 1]),
                                               sqrt(all_store_data[num_epoch-1, 2]),
                                               [start_time, final_time],
                                               sqrt(all_store_data[num_epoch-1, 2]), img_diff)
    
    time.sleep(5)
    # create report
    VF.create_report(net, stage, model_specification)



# ============================== TESTING ======================================
else:
    
    # read model
    val_net = torch.load('./RESULTS/' + validataion_model)
    test_output = val_net(test_inputs)
    
    
    VF.plot_output(stage, num_epoch, test_labels, test_output, img_size, saveSVG=True)
    

time.sleep(5)
os.rename('RESULTS', 'RESULTS_' + stage)
time.sleep(5)


# clear Variable explorer in Spyder
def __reset__(): get_ipython().magic('reset -sf')

    



