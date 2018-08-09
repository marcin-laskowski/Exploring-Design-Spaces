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
import VisualizationFunction3 as VF
import ModelStage_0 as M0

import warnings
warnings.filterwarnings("ignore")
start_time = time.time()

#from IPython import get_ipython



def network(args, model):
    # ========================= PARAMETERS ========================================
    stage = 'stage_0_0' + args.name
    
    Train = 1                                   # True if you want to train
    
    device_type = args.device_type              # cpu or cuda
    device = torch.device(device_type)
    
    learning_rate = args.lr                     # 0.001
    num_epoch = args.epochs                     # 1000
    mini_batch_size = args.batch                # 50
    momentum = args.momentum                    # 0.9
    img_size = 64                               # 64
    num_channels = 1                            # 1
    
    split = args.split                          # 6/10 for 5000 epochs
    
    validation_model = '.pt'                    # load mod


    # ==================== INPUT DATA =============================================
    Labels = np.load('./DATA/05_noPressBEST/Labels_DataSet.npy')
    Inputs = Labels
    
    # use part of the input data
    Inputs = Inputs[:1000]
    Labels = Labels[:1000]
    
    # check data
    #VF.plot_sample_param_and_label(Inputs, Labels)
    
    # prepare data and split for training and testing
    train_inputs, train_labels, test_inputs, test_labels = VF.load_data(Inputs, Labels, split, mini_batch_size, device, img_size)
    print('PROGRESS: Data Prepared!')


    # calcualte time
    start_time = 0
    start_time = time.time()
    
    
    # calling the model
    net = model.to(device)    
    
    criterion = nn.MSELoss().to(device)
    #        criterion = nn.BCELoss()
    #        criterion = nn.PoissonNLLLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
    #        optimizer = optim.SGD(params=net.parameters(), lr=learning_rate, momentum=momentum)
    
    params = list(net.parameters())
    
    all_store_data = np.zeros((num_epoch, 3))



    # ============================== TRAINING =====================================
    if Train == True:
        
        for epoch in range(num_epoch):

            for i in range(0, train_inputs.size(0), mini_batch_size):

                # training data
                train_input = train_inputs[i:i+mini_batch_size, :, :, :]  # (:, 1, 5, 6)
                train_label = train_labels[i:i+mini_batch_size, :, :, :]  # (:, 1, 64, 64)
    
                # forward path
                train_out = net(train_input)
    
                loss = criterion(train_out, train_label)
                loss.backward()
                optimizer.step()
    
                net.zero_grad()
    
    
            end_time = time.time()
    
            # get training data
            epoch_data, train_outputs, test_outputs = VF.epoch_progress(epoch, num_epoch, mini_batch_size, net, train_inputs, train_labels,
                                                                        test_inputs, test_labels, criterion, start_time,
                                                                        end_time, img_size, device)

            # store all training data
            all_store_data[epoch, :] = epoch_data
    
        
        # stop training time
        final_time = time.time()
        print('PROGRESS: Training Done!')
        
        # save output
        VF.save_data(train_outputs, 'train_outputs')
        VF.save_data(test_outputs, 'test_outputs')
    
        # plot results
        VF.result_plot(stage, all_store_data, criterion, True, saveSVG=True)
        VF.plot_output(stage, num_epoch, test_labels, test_outputs, img_size, True, saveSVG=True)
        VF.plot_max_stress(test_labels, test_outputs, stage, all_store_data, criterion, True, saveSVG=True)
        VF.draw_stress_propagation_plot(test_labels[0:4], test_outputs[0:4], stage, all_store_data, criterion, True, saveSVG=True)
                                        
        # save model and state_dict
        VF.save_model(net, stage)
    
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
        
        print('PROGRESS: Data Saved!')
        time.sleep(2)
    
        # create report
        VF.create_report(net, stage, model_specification)
        VF.create_report_page2(net, stage, model_specification)
        print('PROGRESS: Report Created!')
    
    
    
    # ============================== TESTING ======================================
    else:
    
        # read model
        val_net = torch.load(validataion_model)
        test_output = val_net(test_inputs)
        print('PROGRESS: Validation Done!')
    
        VF.plot_output(stage, num_epoch, test_labels, test_output, img_size, saveSVG=True)
        print('PROGRESS: DONE!')
    
    
    # rename folder
    time.sleep(2)
    time_data = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    os.rename('RESULTS', 'RESULTS_' + stage + time_data)
    
    time.sleep(1)
    print('PROGRESS: DONE!')
    
    os.remove('report_plot.png')
    os.remove('report_output.png')
    os.remove('report_maxstress.png')
    os.remove('report_stressprop.png')