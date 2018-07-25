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

#from IPython import get_ipython




for iteration in range(2):

    if iteration == 0:
        Inputs = np.load('./DATA/02_data_diffShape/Images_DataSet.npy')
        Labels = np.load('./DATA/02_data_diffShape/Images_DataSet.npy')
    else:
        Inputs = np.load('./DATA/02_data_diffShape/Labels_DataSet.npy')
        Labels = np.load('./DATA/02_data_diffShape/Labels_DataSet.npy')


    for iteration_2 in range(5,6):


        start_time = 0
        start_time = time.time()

        # -----------------------------------------------------------------------------
        stage = 'Stage_0_' + str(iteration) + '_' + str(iteration_2)
        print('/\/\/\/\/\/\/\/\/\/\/\ ' + stage + ' /\/\/\/\/\/\/\/\/\/\/\/')


        # ========================= PARAMETERS ========================================
        Train = 1

        device_type = 'cuda'
        learning_rate = 0.001
        num_epoch = 200
        mini_batch_size = 100
        momentum = 0.9
        img_size = 64
        num_channels = 1

        split = 7/8
        
        validation_model = '.pt'

        # ==================== INPUT DATA =============================================
        device = torch.device(device_type)  # cpu or cuda

        Inputs = Inputs[:8000]
        Labels = Labels[:8000]
        train_inputs, train_labels, test_inputs, test_labels = VF.load_data(Inputs, Labels, split, mini_batch_size, device, img_size)


        # ====================== NEURAL NETWORK CONFIGURATION =========================
        # =================================================================================
        # =================================================================================

        if iteration_2 == 0:
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.h1 = nn.Linear(64*64, 300)
                    self.h2 = nn.Linear(300, 30)
                    self.h3 = nn.Linear(30, 300)
                    self.h4 = nn.Linear(300, 64*64)

                def forward(self, x):
                    x = x.view(x.size(0), 64*64)
                    x = F.relu(self.h1(x))
                    x = F.relu(self.h2(x))
                    x = F.relu(self.h3(x))
                    x = F.relu(self.h4(x))
                    return x


        elif iteration_2 == 1:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(64*64, 200),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(200, 40),
                        nn.ReLU())
                    self.decoder = nn.Sequential(
                        nn.Linear(40,200),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(200, 64*64),
                        nn.RReLU(0, 1))

                def forward(self, x):
                    x = x.view(x.size(0), 64*64)
                    x = self.encoder(x)
                    x = self.decoder(x)
                    return x


        elif iteration_2 == 2:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.encoder = nn.Sequential(
                        nn.Conv2d(1, 6, kernel_size=6, stride=2, padding=0),
                        nn.ReLU(True),
                        nn.Conv2d(6, 16, kernel_size=6, stride=2, padding=0),
                        nn.ReLU(True),  # output: 1, 16, 13, 13
                        )
                    self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(16, 6, 6, stride=2),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(6, 1, 6, stride=2),
                        nn.RReLU(0, 1)
                        )

                def forward(self, x):
                    x = self.encoder(x)
                    x = self.decoder(x)
                    return x


        elif iteration_2 == 3:
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
                        nn.Sigmoid())

                def forward(self, x):
                    x = self.encoder(x)
                    x = self.decoder(x.view(x.size(0), 16, 10, 10))
                    return x


        elif iteration_2 == 4:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.encoder1 = nn.Sequential(
                        nn.Conv2d(1, 6, kernel_size=6, stride=2, padding=0),
                        nn.ReLU(True),
                        nn.Conv2d(6, 16, kernel_size=6, stride=2, padding=0),
                        nn.ReLU(True))  # output: 1, 16, 13, 13
                    self.encoder2 = nn.Sequential(
                        nn.Linear(16 * 13 * 13, 120),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(120, 40),
                        nn.ReLU())
                    self.decoder1 = nn.Sequential(
                        nn.Linear(40, 120),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(120, 16 * 13 * 13),
                        nn.ReLU())
                    self.decoder2 = nn.Sequential(
                        nn.ConvTranspose2d(16, 6, 6, stride=2),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(6, 1, 6, stride=2),
                        nn.Sigmoid())    
                    
                def forward(self, x):
                    x = self.encoder1(x)
                    x = self.encoder2(x.view(x.size(0), 16 * 13 * 13))
                    x = self.decoder1(x)
                    x = self.decoder2(x.view(x.size(0), 16, 13, 13))
                    return x


        elif iteration_2 == 5:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.encoder1 = nn.Sequential(
                        nn.Conv2d(1, 6, kernel_size=6, stride=2, padding=0),
                        nn.ReLU(True),
                        nn.MaxPool2d(kernel_size=3, stride=1),
                        nn.Conv2d(6, 16, kernel_size=6, stride=2, padding=0),
                        nn.ReLU(True),
                        nn.MaxPool2d(kernel_size=3, stride=1))  # output: 1, 16, 10, 10
                    self.encoder2 = nn.Sequential(
                        nn.Linear(16 * 10 * 10, 500),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(500, 40),
                        nn.ReLU())
                    self.decoder1 = nn.Sequential(
                        nn.Linear(40, 500),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(500, 16 * 10 * 10),
                        nn.ReLU())
                    self.decoder2 = nn.Sequential(
                        nn.ConvTranspose2d(16, 10, 6, stride=2),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(10, 6, 6, stride=2),
                        nn.ReLU(),
                        nn.ConvTranspose2d(6, 3, 6, stride=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(3, 1, 8, stride=1),
                        nn.Sigmoid())   
                    
                def forward(self, x):
                    x = self.encoder1(x)
                    x = self.encoder2(x.view(x.size(0), 16 * 10 * 10))
                    x = self.decoder1(x)
                    x = self.decoder2(x.view(x.size(0), 16, 10, 10))
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
            VF.result_plot('loss_' + stage, all_store_data, criterion, True, saveSVG=True)
            VF.plot_output('out_' + stage, num_epoch, test_labels, test_outputs, img_size, True, saveSVG=True)

            # save model and state_dict
            VF.save_model(net, 'Autoencoder_' + str(iteration_2))

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


            VF.plot_output(stage, num_epoch, test_labels, test_output, img_size, saveSVG=True)


        time.sleep(5)
        os.rename('RESULTS', 'RESULTS_' + str(iteration) + '_' + str(iteration_2))
        time.sleep(5)

        # clear Variable explorer in Spyder
#        def __reset__(): get_ipython().magic('reset -sf')
