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
import VisualizationFunction2 as VF

import warnings
warnings.filterwarnings("ignore")
start_time = time.time()

# from IPython import get_ipython




# ========================= PARAMETERS ========================================
Train = 1

device_type = 'cuda'
learning_rate = 0.001       # 0.001
num_epoch = 5
mini_batch_size = 20       # 100
momentum = 0.9
img_size = 64
num_channels = 1

split = 4/5              # 6/10, 5000


device = torch.device(device_type)  # cpu or cuda

validation_model = '.pt'


# =========================== PREPARE DATA ====================================

#number_of_elements = 10000
#points_and_fix = GDF.automate_get_params(number_of_elements)
#
#input_data = torch.zeros((number_of_elements, 5, 6))
#input_data = input_data.float()
#for i in range(number_of_elements):
#    input_data[i] = torch.from_numpy(points_and_fix[i])


# ==================== INPUT DATA =============================================
#Inputs = input_data.view(input_data.size(0), 1, 1, 5, 6)
#Inputs = torch.from_numpy(np.load('./DATA/03_data_new/InputParams_matrix.npy'))
#Inputs = Inputs.view(Inputs.size(0), 1, 1, 5, 6)
#Labels = np.load('./DATA/03_data_new/Labels_DataSet.npy')
Inputs = torch.from_numpy(np.load('./DATA/04_diffShapeBEST/Params_DataSet.npy'))
Inputs = Inputs.view(Inputs.size(0), 1, 1, 5, 6)
Labels = np.load('./DATA/04_diffShapeBEST/Labels_DataSet.npy')

#Inputs = Inputs[:5000]
#Labels = Labels[:5000]

# check data
#VF.plot_sample_param_and_label(Inputs, Labels)

train_inputs, train_labels, test_inputs, test_labels = VF.load_data(Inputs, Labels, split, mini_batch_size, device, img_size)


print('Data Prepared!')


for iteration in range(1):

    for iteration_2 in range(1,3):

        start_time = 0
        start_time = time.time()

        # -----------------------------------------------------------------------------
        stage = 'Stage_1_3_' + str(iteration) + '_' + str(iteration_2)
        print('====================== ' + stage + ' =======================')


        # ====================== NEURAL NETWORK CONFIGURATION =========================
        # =================================================================================
        # =================================================================================

        if iteration_2 == 0:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
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


        elif iteration_2 == 1:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
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



        elif iteration_2 == 2:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
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

        else:
            print("WRONG MODEL!")


        # =================================================================================
        # =================================================================================
        # =================================================================================



        # calling the model
        net = Net().to(device)

        criterion = nn.MSELoss().to(device)
#        criterion = nn.BCELoss()
#        criterion = nn.PoissonNLLLoss()
        optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
#        optimizer = optim.SGD(params=net.parameters(), lr=learning_rate, momentum=momentum)

        params = list(net.parameters())
        loss_sum = 0
        best_test_value = 0

        train_test_loss = np.zeros((num_epoch, 2))
        all_store_data = np.zeros((num_epoch, 3))

        # ============================== TRAINING =====================================
        if Train == True:

            print('Lets start Training!')
            learning_rate = 0.001

            # Learning process
            for epoch in range(num_epoch):

                # adjust learning rate
                if epoch%10 == 0:
                    learning_rate = learning_rate * 0.98

                running_loss = 0.0

                for i in range(0, train_inputs.size(0), mini_batch_size):

                    # -------------------- TRAIN DATA ---------------------------------
                    train_input = train_inputs[i:i+mini_batch_size, :, :, :]  # train_input.size --> (:, 1, 64, 64)
                    train_label = train_labels[i:i+mini_batch_size, :, :, :]  # train_labels.size --> (:, 1, 64, 64)

                    # forward path
                    train_out = net(train_input)

                    loss = criterion(train_out, train_label)
                    loss.backward()
                    optimizer.step()
#                    running_loss += loss.data[0]
#                    loss_sum += loss.data[0]

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
            VF.save_model(net, stage + '_' + str(iteration_2))

            # obtain image difference
            img_diff = VF.image_difference(test_labels, test_outputs)

            time.sleep(5)

            # save all parameters
            model_specification = VF.net_specification(stage, train_input.size(0),
                                                       str(train_inputs.size(2)) +' x ' + str(train_inputs.size(3)),
                                                       str(train_labels.size(2)) +' x ' + str(train_labels.size(3)),
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
        os.rename('RESULTS', 'RESULTS_' + stage )
        time.sleep(2)

        os.remove('report_plot.png')
        os.remove('report_output.png')


        # clear Variable explorer in Spyder
        # def __reset__(): get_ipython().magic('reset -sf')
