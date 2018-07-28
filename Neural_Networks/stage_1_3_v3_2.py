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

device_type = 'cpu'
num_epoch = 1000
img_size = 64
num_channels = 1

split = 4/5              # 6/10, 5000


device = torch.device(device_type)  # cpu or cuda

validation_model = '.pt'


# =========================== PREPARE DATA ====================================
"""
number_of_elements = 10000
points_and_fix = GDF.automate_get_params(number_of_elements)

input_data = torch.zeros((number_of_elements, 5, 6))
input_data = input_data.float()
for i in range(number_of_elements):
    input_data[i] = torch.from_numpy(points_and_fix[i])



Inputs = input_data.view(input_data.size(0), 1, 1, 5, 6)
#Labels = np.load('./DATA/02_data_diffShape/Labels_DataSet.npy')
#Labels = np.load('./DATA/01_data_noPress/Labels_DataSet.npy')
Labels = np.load('./DATA/03_data_new/Labels_DataSet.npy')
"""

Inputs = torch.from_numpy(np.load('./DATA/04_diffShapeBEST/Params_DataSet.npy'))
Inputs = Inputs.view(Inputs.size(0), 1, 1, 5, 6)
Labels = np.load('./DATA/04_diffShapeBEST/Labels_DataSet.npy')

#Inputs = Inputs[:8000]
#Labels = Labels[:8000]

print('Data Prepared!')





for iteration in range(2):

    for iteration_2 in range(1,2):
        
        
        # ======================== HYPERPARAMETERS ===================================
        learning_rate = 0.001       # 0.001
        momentum = 0.9
        if iteration_2 == 0:
            mini_batch_size = 5
        elif iteration_2 == 1:
            mini_batch_size = 20
        elif iteration_2 == 2:
            mini_batch_size = 30
        else:
            mini_batch_size = 50
        
        
        # split your data
        train_inputs, train_labels, test_inputs, test_labels = VF.load_data(Inputs, Labels, split, mini_batch_size, device, img_size)
        

        start_time = 0
        start_time = time.time()
        
        # stage
        stage = 'Stage_1_3_' + str(iteration) + '_' + str(iteration_2)
        print('====================== ' + stage + ' =======================')


        # ====================== NEURAL NETWORK CONFIGURATION =========================
        
        if iteration == 0:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.converter= nn.Sequential(
                        nn.Linear(5*6, 100),
                        nn.ReLU(),
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
                    x = self.decoder(x.view(x.size(0), 20, 3, 3))
                    x = x.view(x.size(0), 1, 64, 64)
                    return x
                
        elif iteration == 1:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.converter= nn.Sequential(
                        nn.Linear(5*6, 200),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(200, 500),
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
                    x = self.decoder(x.view(x.size(0), 25, 3, 3))
                    x = x.view(x.size(0), 1, 64, 64)
                    return x


        elif iteration == 2:
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
                        nn.Dropout(0.2),
                        nn.Linear(1200, 50 * 4 * 4),
                        nn.ReLU())
                    self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
                        nn.ConvTranspose2d(50,20,2,2,0,0),
                        nn.BatchNorm2d(20),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(20,10,2,2,0,0),
                        nn.BatchNorm2d(10),
                        nn.ReLU(),
                        nn.ConvTranspose2d(10,5,2,2,0,0),
                        nn.BatchNorm2d(5),
                        nn.ReLU(),
                        nn.ConvTranspose2d(5,1,2,2,0,0),
                        nn.ReLU())
    
                def forward(self, x):
                    x = x.view(x.size(0), 5*6)
                    x = self.converter(x)
                    x = self.decoder(x.view(x.size(0), 50, 4, 4))
                    x = x.view(x.size(0), 1, 64, 64)
                    return x
                
        elif iteration == 3:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.converter= nn.Sequential(
                        nn.Linear(5*6, 100),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(100, 400),
                        nn.ReLU(True),
                        nn.Dropout(0.2),
                        nn.Linear(400, 600),
                        nn.ReLU(True),
                        nn.Dropout(0.2),
                        nn.Linear(600, 50 * 4 * 4),
                        nn.ReLU())
                    self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
                        nn.ConvTranspose2d(50,20,2,2,0,0),
                        nn.BatchNorm2d(20),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(20,10,2,2,0,0),
                        nn.BatchNorm2d(10),
                        nn.ReLU(),
                        nn.ConvTranspose2d(10,5,2,2,0,0),
                        nn.BatchNorm2d(5),
                        nn.ReLU(),
                        nn.ConvTranspose2d(5,1,2,2,0,0),
                        nn.ReLU())
    
                def forward(self, x):
                    x = x.view(x.size(0), 5*6)
                    x = self.converter(x)
                    x = self.decoder(x.view(x.size(0), 50, 4, 4))
                    x = x.view(x.size(0), 1, 64, 64)
                    return x
                

        elif iteration == 4:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.converter= nn.Sequential(
                        nn.Linear(5*6, 60),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(60, 180),
                        nn.ReLU(True),
                        nn.Dropout(0.2),
                        nn.Linear(180, 400),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(400, 50 * 4 * 4),
                        nn.ReLU())
                    self.decoder = nn.Sequential(  # x - f + 2p / s = x_out
                        nn.ConvTranspose2d(50,20,2,2,0,0),
                        nn.BatchNorm2d(20),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(20,10,2,2,0,0),
                        nn.BatchNorm2d(10),
                        nn.ReLU(),
                        nn.ConvTranspose2d(10,5,2,2,0,0),
                        nn.BatchNorm2d(5),
                        nn.ReLU(),
                        nn.ConvTranspose2d(5,1,2,2,0,0),
                        nn.ReLU())
    
                def forward(self, x):
                    x = x.view(x.size(0), 5*6)
                    x = self.converter(x)
                    x = self.decoder(x.view(x.size(0), 50, 4, 4))
                    x = x.view(x.size(0), 1, 64, 64)
                    return x
            
            
        else:
            print('WRONG MODEL!')
                
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

            print('Lets start Training!')
            learning_rate = 0.001

            # Learning process
            for epoch in range(num_epoch):

                # adjust learning rate
                if iteration == 0:
                    if epoch%10 == 0:
                        learning_rate = learning_rate * 0.98
                else:
                    pass

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
            VF.save_model(net, stage)

            # obtain image difference
            img_diff = VF.image_difference(test_labels, test_outputs)
            time.sleep(3)

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

            time.sleep(3)
            
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

#        os.remove('report_plot.png')
#        os.remove('report_output.png')


        # clear Variable explorer in Spyder
        # def __reset__(): get_ipython().magic('reset -sf')
