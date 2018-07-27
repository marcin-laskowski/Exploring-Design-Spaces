import os
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import time
import datetime
from math import sqrt

import GenerateDataFunctions as GDF
import VisualizationFunction2_2 as VF

import warnings
warnings.filterwarnings("ignore")
start_time = time.time()

#from IPython import get_ipython




# ========================= PARAMETERS ========================================
Train = 1

device_type = 'cuda'
learning_rate = 0.001
num_epoch = 5000
mini_batch_size = 50
momentum = 0.9
img_size = 64
num_channels = 1

split = 5/6


device = torch.device(device_type)  # cpu or cuda

validation_model = '.pt'



# =========================== PREPARE DATA ====================================

number_of_elements = 10000
fix_and_force = GDF.automate_get_fix_and_force(number_of_elements)

input_data = torch.zeros((number_of_elements, 5, 4))
input_data = input_data.float()
for i in range(number_of_elements):
    input_data[i] = torch.from_numpy(fix_and_force[i])
    

Inputs = np.load('./DATA/03_data_new/Images_DataSet.npy')
Labels = np.load('./DATA/03_data_new/Labels_DataSet.npy')
Params = input_data.view(input_data.size(0), 1, 1, 5, 4)

Inputs = Inputs[:6000]
Labels = Labels[:6000]
Params = Params[:6000]

train_inputs, train_labels, test_inputs, test_labels = VF.load_data(Inputs, Labels, split, mini_batch_size, device, img_size)
train_params, test_params = VF.load_params(Params, split, mini_batch_size, device, img_size)



# ========================== MODEL ============================================

for iteration in range(1):

    for iteration_2 in range(1,2):

        start_time = 0
        start_time = time.time()

        # -----------------------------------------------------------------------------
        stage = 'Stage_1_2_' + str(iteration) + '_' + str(iteration_2)
        
        print('===================== ' + stage + ' =======================')


        # ====================== NEURAL NETWORK CONFIGURATION =========================
        # =================================================================================
        # =================================================================================

        if iteration_2 == 0:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
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
                

        elif iteration_2 == 1:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
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
                        nn.Linear(40+20, 500),
                        nn.ReLU(True),
                        nn.Dropout(0.2),
                        nn.Linear(500, 500),
                        nn.ReLU(True),
                        nn.Dropout(0.5),
                        nn.Linear(500, 200),
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
                        nn.ConvTranspose2d(5,1,2,2,0,0),
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
            

        else:
            print("WRONG MODEL!")


        # =================================================================================
        # =================================================================================
        # =================================================================================



        # calling the model
        net = Net().to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
#        optimizer = optim.SGD(params=net.parameters(), lr=learning_rate, momentum=momentum)

        params = list(net.parameters())
        loss_sum = 0
        best_test_value = 0

        train_test_loss = np.zeros((num_epoch, 2))
        all_store_data = np.zeros((num_epoch, 3))

        # ============================== TRAINING =====================================
        if Train == True:
            # Learning process
            for epoch in range(num_epoch):

                running_loss = 0.0
                
                for i in range(0, train_inputs.size(0), mini_batch_size):

                    # -------------------- TRAIN DATA ---------------------------------
                    train_input = train_inputs[i:i+mini_batch_size, :, :, :]  # train_input.size --> (:, 1, 64, 64)
                    train_label = train_labels[i:i+mini_batch_size, :, :, :]  # train_labels.size --> (:, 1, 64, 64)
                    train_param = train_params[i:i+mini_batch_size, :, :, :]
                    
                    # forward path
                    train_out = net(train_input, train_param)

                    loss = criterion(train_out, train_label)
                    loss.backward()
                    optimizer.step()
#                    running_loss += loss.data[0]
#                    loss_sum += loss.data[0]

                    net.zero_grad()




                end_time = time.time()

                epoch_data, train_outputs, test_outputs = VF.epoch_progress(epoch, num_epoch, mini_batch_size, net, train_inputs, train_labels, train_params,
                                                                            test_inputs, test_labels, test_params, criterion, start_time,
                                                                            end_time, img_size, device)

                all_store_data[epoch, :] = epoch_data

            final_time = time.time()

            # plot results
            VF.result_plot('loss_' + stage, all_store_data, criterion, True, saveSVG=True)
            VF.plot_output('out_' + stage, num_epoch, test_labels, test_outputs, img_size, True, saveSVG=True)

            # save model and state_dict
            VF.save_model(net, 'Stage_1_2_' + str(iteration_2))

            # obtain image difference
            img_diff = VF.image_difference(test_labels, test_outputs)

            time.sleep(5)

            # save all parameters
            model_specification = VF.net_specification(stage, train_input.size(0),
                                                       str(train_inputs.size(2)) +' x ' + str(train_inputs.size(3)),
                                                       str(test_inputs.size(2)) +' x ' + str(test_inputs.size(3)),
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


            VF.plot_output('out_' + stage, num_epoch, test_labels, test_output, img_size, saveSVG=True)


        time.sleep(2)
        os.rename('RESULTS', stage)
        time.sleep(2)

        # clear Variable explorer in Spyder
#        def __reset__(): get_ipython().magic('reset -sf')
