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
import VisualizationFunction as VF

import warnings
warnings.filterwarnings("ignore")
start_time = time.time()

from IPython import get_ipython




# ========================= PARAMETERS ========================================
Train = 1

device_type = 'cpu'
learning_rate = 0.001
num_epoch = 10
mini_batch_size = 200
momentum = 0.9
img_size = 64
num_channels = 1

split = 5/8


device = torch.device(device_type)  # cpu or cuda

validation_model = '.pt'



# =========================== PREPARE DATA ====================================

number_of_elements = 10000
fix_and_force = GDF.automate_get_fix_and_force(number_of_elements)

input_data = torch.zeros((number_of_elements, 5, 6))
input_data = input_data.float()
for i in range(number_of_elements):
    input_data[i] = torch.from_numpy(fix_and_force[i])
    

Inputs = np.load('./DATA/02_data_diffShape/Images_DataSet.npy')
Labels = np.load('./DATA/02_data_diffShape/Labels_DataSet.npy')
Params = input_data.view(input_data.size(0), 1, 1, 5, 4)

Inputs = Inputs[:8000]
Labels = Labels[:8000]
Params = Params[:8000]

train_inputs, train_labels, test_inputs, test_labels = VF.load_data(Inputs, Labels, split, mini_batch_size, device, img_size)
train_params, test_params = VF.load_params(Params, split, mini_batch_size, device, img_size)



# ========================== MODEL ============================================

for iteration in range(1):

    for iteration_2 in range(1):

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
                        nn.Linear(64*64, 500),
                        nn.ReLU(True),
                        nn.Linear(500, 100),
                        nn.ReLU(True),
                        nn.Linear(100, 40),
                        nn.ReLU(True))
                    self.converter = nn.Sequential(
                        nn.Linear(60, 100),
                        nn.ReLU(True),
                        nn.Linear(100, 500),
                        nn.ReLU(True),
                        nn.Linear(500, 16*13*13),
                        nn.ReLU(True),
                        nn.Dropout())
                    self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
                        nn.ConvTranspose2d(16, 6, 6, 2),
                        nn.BatchNorm2d(6),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(6, 1, 6, 2),
                        nn.BatchNorm2d(1),
                        nn.ReLU())
            
                def forward(self, x, y):
                    x = x.view(x.size(0), 64*64)
                    x = self.encoder(x)
                    x = x.view(x.size(0), 20)
                    y = y.view(y.size(0), 20)
                    x = torch.cat((x, y), 1)
                    x = self.converter(x)
                    x = self.decoder(x.view(x.size(0), 16, 13, 13))
                    x = x.view(x.size(0), 1, 64, 64)
                    return x


        elif iteration_2 == 1:
            print("WRONG MODEL!")
            

        else:
            print("WRONG MODEL!")


        # =================================================================================
        # =================================================================================
        # =================================================================================



        # calling the model
        net = Net().to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

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

                for i in range(mini_batch_size):

                    # -------------------- TRAIN DATA ---------------------------------
                    train_input = train_inputs[:, i, :, :, :]  # train_input.size --> (:, 1, 64, 64)
                    train_label = train_labels[:, i, :, :, :]  # train_labels.size --> (:, 1, 64, 64)
                    train_params = train_params[:, i, :, :, :]

                    # forward path
                    train_out = net(train_input, train_params)

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
            VF.result_plot('loss_' + stage, all_store_data, criterion, True, saveSVG=True)
            VF.plot_output('out_' + stage, num_epoch, test_labels, test_outputs, img_size, True, saveSVG=True)

            # save model and state_dict
            VF.save_model(net, 'Autoencoder_' + str(iteration_2))

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


            VF.plot_output('out_' + stage, num_epoch, test_labels, test_output, img_size, saveSVG=True)


        time.sleep(5)
        os.rename('RESULTS', 'RESULTS_' + str(iteration) + '_' + str(iteration_2))
        time.sleep(5)

        # clear Variable explorer in Spyder
        def __reset__(): get_ipython().magic('reset -sf')
