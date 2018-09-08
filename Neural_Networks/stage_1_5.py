import os
import argparse

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
import ModelStage_1_5 as M15

import warnings
warnings.filterwarnings("ignore")
start_time = time.time()

# from IPython import get_ipython


def network(args, model):
    # ========================= PARAMETERS ========================================
    stage = 'stage_1_5_' + args.name
    
    Train = 1                                   # True if you want to train
    
    device_type = args.device_type              # cpu or cuda
    device = torch.device(device_type)
    
    dataset = args.dataset
    
    learning_rate = args.lr                     # 0.001
    num_epoch = args.epochs                     # 1000
    mini_batch_size = args.batch                # 50
    momentum = args.momentum                    # 0.9
    img_size = 64                               # 64
    num_channels = 3                            # 3
    
    split = args.split                          # 6/10 for 5000 epochs
    
    validation_model = '.pt'                    # load model
    validation_model_state_dict = '.pt'             # load model state_dict
    
    
    # ==================== INPUT DATA =============================================
    Inputs = torch.from_numpy(np.load('./DATA/' + dataset + '/Inputs_3.npy'))
    Labels = torch.from_numpy(np.load('./DATA/' + dataset + '/Labels_3.npy'))
    Params = torch.from_numpy(np.load('./DATA/' + dataset + '/Params_DataSet.npy'))
#    Params = VF.combine_fix_and_force(Params)
#    Params = Params.view(Params.size(0), 1, 5, 4)
    Params = Params.view(Params.size(0), 1, 5, 6)
    Params[:, :, :, 4:6] = Params[:, :, :, 4:6] / 10
    
    # use part of the input data
#    Inputs = Inputs[:1000]
#    Labels = Labels[:1000]
#    Params = Params[:1000]
    
    # check data
    #VF.plot_sample_param_and_label(Inputs, Labels)
    
    # prepare data and split for training and testing
    _, train_labels, _, test_labels = VF.load_data(Inputs, Labels, split, mini_batch_size, device, img_size)
    train_inputs, test_inputs = VF.load_params(Params, split, mini_batch_size, device, img_size)

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
    best_test_loss_value = 10000
    smallest_test_loss = 1000
    n = 0
    
    # ============================== TRAINING =====================================
    if Train == True:
    
        # Learning process
        for epoch in range(num_epoch):
    
            for i in range(0, train_inputs.size(0), mini_batch_size):
                # training data
                train_input = train_inputs[i:i+mini_batch_size, :, :, :]  # (:, 1, 5, 6)
                train_label = train_labels[i:i+mini_batch_size, :, :, :]  # (:, 3, 64, 64)

                # forward path
                train_out = net(train_input)
                
                loss = criterion(train_out, train_label)
                loss.backward()
                optimizer.step()
    
                net.zero_grad()
 
            end_time = time.time()
    
            # get training data
            epoch_data = VF.epoch_progress(epoch, num_epoch, mini_batch_size, net, train_inputs, train_labels,
                                           test_inputs, test_labels, criterion, start_time,
                                           end_time, img_size, device)
    
            # store all training data
            all_store_data[epoch, :] = epoch_data

            # save best model
            best_test_loss_value = VF.get_best_model(net, stage, best_test_loss_value, all_store_data[epoch, 2])


            # early stoping
            smallest_test_loss, n = VF.early_stop(epoch, epoch_data[0,2], smallest_test_loss, n)
            
            if n == 50:
                num_epoch = epoch + 1
                break    

        # save all_store_data
        VF.save_data(all_store_data, 'all_store_data_' + stage)

        # stop training time
        final_time = time.time()
        print('PROGRESS: Training Done!')
        
        # get train_outputs and train_final_loss
        train_total_loss, train_outputs = VF.get_loss_and_output(net, train_inputs, train_labels, mini_batch_size, criterion, device)
        
        # get test_outputs and test_final_loss
        test_total_loss, test_outputs = VF.get_loss_and_output(net, test_inputs, test_labels, mini_batch_size, criterion, device)
        print('PROGRESS: Outputs Obtained!')
            
        
        # save output
        VF.save_data(train_outputs, 'train_outputs')
        VF.save_data(test_outputs, 'test_outputs')
    
        # plot results
        VF.result_plot(stage, all_store_data, criterion, True, saveSVG=True)
        VF.plot_output('input' + stage, num_epoch, train_labels, train_outputs, img_size, False, saveSVG=True)
        VF.plot_output(stage, num_epoch, test_labels, test_outputs, img_size, True, saveSVG=True)
        VF.plot_max_stress(test_labels, test_outputs, stage, all_store_data, criterion, True, saveSVG=True)
        VF.draw_stress_propagation_plot(test_labels[0:4], test_outputs[0:4], stage, all_store_data, criterion, True, saveSVG=True)
        VF.mean_and_CI(test_labels, test_outputs, stage, all_store_data, criterion, True, saveSVG=False)
                                        
        # save model and state_dict
        VF.save_model(net, stage)
    
        # obtain image difference
        img_diff = VF.image_difference(test_labels, test_outputs)
        
        # calculate max stress error
        overall_max_stress_error = VF.avg_max_stress_error(test_labels, test_outputs, img_size)
    
        time.sleep(2)
    
        # save all parameters
        model_specification = VF.net_specification(stage, str(train_inputs.size(0) + test_inputs.size(0)),
                                                   str(train_inputs.size(2)) +' x ' + str(train_inputs.size(3)),
                                                   str(train_labels.size(2)) +' x ' + str(train_labels.size(3)),
                                                   num_epoch, mini_batch_size, learning_rate, momentum, str(criterion),
                                                   optimizer, str(train_inputs.size(0)) + ' / ' + str(test_inputs.size(0)),
                                                   device_type, np.min(all_store_data[:,1]), np.max(all_store_data[:,1]),
                                                   np.min(all_store_data[:,2]), np.max(all_store_data[:,2]),
                                                   train_total_loss,
                                                   test_total_loss,
                                                   [start_time, final_time],
                                                   sqrt(all_store_data[num_epoch-1, 2]), img_diff, dataset,
                                                   overall_max_stress_error)
        
        print('PROGRESS: Data Saved!')
        time.sleep(2)
    
        # create report
        VF.create_report(net, stage, model_specification)
        VF.create_report_page2(net, stage, model_specification)
        print('PROGRESS: Report Created!')
    
    
    
    # ============================== TESTING ======================================
    else:
    
        # read model
#        val_net = torch.load(validataion_model)
        val_net = model
        val_net.state_dict = validation_model_state_dict
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
    os.remove('report_stresspropall.png')
    
    # clear Variable explorer in Spyder
    # def __reset__(): get_ipython().magic('reset -sf')
