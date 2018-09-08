import os
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import itertools

import time
import datetime
from math import sqrt
import sys

import GenerateDataFunctions as GDF
import VisualizationFunction2 as VF
import ModelStage_1_4 as M14

import warnings
warnings.filterwarnings("ignore")
start_time = time.time()

#from IPython import get_ipython


def network(args, Generator_AB, Generator_BA, Discriminator):
    # ========================= PARAMETERS ========================================
    stage = 'stage_1_6_' + args.name
    
    Train = 1                                       # True if you want to train
    
    device_type = args.device_type                  # cpu or cuda
    device = torch.device(device_type)
    
    dataset = args.dataset

    epoch = 0
    num_epoch = args.epochs                         # 1000 
    learning_rate = args.lr                         # 0.001
    mini_batch_size = args.batch                    # 50
    momentum = args.momentum                        # 0.9
    img_size = 64                                   # 64
    num_channels = 3                                # 3
    
    checkpoint_interval = 1
    
    b1 = 0.5
    b2 = 0.999
    
    split = args.split                              # 6/10 for 5000 epochs
    
    validation_model = '.pt'                        # load model
    validation_model_state_dict = '.pt'             # load model state_dict

    
    
   
    
    # ==================== INPUT DATA =============================================
    Inputs = np.load('./DATA/' + dataset + '/Inputs_3.npy')
    Labels = np.load('./DATA/' + dataset + '/Labels_3.npy')
    Params = np.load('./DATA/' + dataset + '/Params_DataSet.npy')
#    Params = VF.combine_fix_and_force(Params)
    
    # use part of the input data
#    Inputs = Inputs[:1000]
#    Labels = Labels[:1000]
#    Params = Params[:1000]
  
    # check data
    #VF.plot_sample_param_and_label(Inputs, Labels)
    
    
    # prepare data and split for training and testing
    train_inputs, train_labels, test_inputs, test_labels = VF.load_data(Inputs, Labels, split, mini_batch_size, device, img_size)
    train_params, test_params = VF.load_params(Params, split, mini_batch_size, device, img_size)
    print('PROGRESS: Data Prepared!')

    
    # ==============================================================================
    
    # Calculate output of image discriminator (PatchGAN)
    patch = (1, img_size//2**3, img_size//2**3)

    # calcualte time
    start_time = 0
    start_time = time.time()


    # Create sample and checkpoint directories
    os.makedirs('./RESULTS/images/%s' % stage, exist_ok=True)
    os.makedirs('./RESULTS/saved_models/%s' % stage, exist_ok=True)

    # initialize weights    
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
            
    
    # ======================== CREATE MODEL ========================================
   
    # Initialize generator and discriminator
    G_AB = Generator_AB.to(device)       # add force and fixation
    G_BA = Generator_BA.to(device)
    D_A = Discriminator.to(device)
    D_B = Discriminator.to(device)

    # Save model
    if epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load('./RESULTS/saved_models/%s/G_AB_%d.pth' % (stage, epoch)))
        G_BA.load_state_dict(torch.load('./RESULTS/saved_models/%s/G_BA_%d.pth' % (stage, epoch)))
        D_A.load_state_dict(torch.load('./RESULTS/saved_models/%s/D_A_%d.pth' % (stage, epoch)))
        D_B.load_state_dict(torch.load('./RESULTS/saved_models/%s/D_B_%d.pth' % (stage, epoch)))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)
    
    
    # ======================== NETWORK SPECIFICATION ===============================
    
    # loss function
    adversarial_loss = torch.nn.MSELoss().to(device)
    cycle_loss = torch.nn.L1Loss().to(device)
    pixelwise_loss = torch.nn.L1Loss().to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=learning_rate, betas=(b1, b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=learning_rate, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=learning_rate, betas=(b1, b2))


    prev_time = time.time()

    all_store_data = np.zeros((num_epoch, 3))

    # ============================== TRAINING =====================================
    if Train == True:
        
        for epoch in range(num_epoch):

            for i in range(0, train_inputs.size(0), mini_batch_size):

                # Model inputs
                real_A = train_inputs[i:i+mini_batch_size, :, :, :]  # (:, 3, 64, 64)
                real_B = train_labels[i:i+mini_batch_size, :, :, :]  # (:, 3, 64, 64)
                real_A_param = train_params[i:i+mini_batch_size, :, :, :]  # (:, 1, 5, 6)
                
                # Adversarial ground truth
                valid = Variable((torch.from_numpy(np.ones((real_A.size(0), *patch)))).float(), requires_grad=False).to(device)
                fake = Variable((torch.from_numpy(np.zeros((real_A.size(0), *patch)))).float(), requires_grad=False).to(device)
                
                
                # --------------------- TRAIN GENERATORS ----------------------
                optimizer_G.zero_grad()
                
                # GAN loss
                fake_B = G_AB(real_A, real_A_param)
                loss_GAN_AB = adversarial_loss(D_B(fake_B), valid)
                fake_A = G_BA(real_B)
                loss_GAN_BA = adversarial_loss(D_A(fake_A), valid)
        
                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        
                # Pixelwise translation loss
                loss_pixelwise = (pixelwise_loss(fake_A, real_A) + \
                                  pixelwise_loss(fake_B, real_B)) / 2
        
                # Cycle loss
                loss_cycle_A = cycle_loss(G_BA(fake_B), real_A)
                loss_cycle_B = cycle_loss(G_AB(fake_A, real_A_param), real_B)
                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        
                # Total loss
                loss_G = loss_GAN + loss_cycle + loss_pixelwise
        
                loss_G.backward()
                optimizer_G.step()
                
                
                # ------------------ TRAIN DISCRIMINATOR A --------------------
                optimizer_D_A.zero_grad()

                # Real loss
                loss_real = adversarial_loss(D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                loss_fake = adversarial_loss(D_A(fake_A.detach()), fake)
                # Total loss
                loss_D_A = (loss_real + loss_fake) / 2
        
                loss_D_A.backward()
                optimizer_D_A.step()              
                
                
                # ------------------ TRAIN DISCRIMINATOR B --------------------
                optimizer_D_B.zero_grad()
                # Real loss
                loss_real = adversarial_loss(D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                loss_fake = adversarial_loss(D_B(fake_B.detach()), fake)
                # Total loss
                loss_D_B = (loss_real + loss_fake) / 2
        
                loss_D_B.backward()
                optimizer_D_B.step()
        
                loss_D = 0.5 * (loss_D_A + loss_D_B)
                
            # ---------------- PRINT PROGRESS -----------------------------
        
            # Determine approximate time left
            time_left = datetime.timedelta(seconds=num_epoch * (time.time() - prev_time))
            prev_time = time.time()
        
            # Print log
            sys.stdout.write("\r[Epoch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f, cycle: %f] ETA: %s" %
                             (epoch, num_epoch, loss_D.item(), loss_G.item(), loss_GAN.item(), loss_pixelwise.item(),
                              loss_cycle.item(), time_left))        
        
            
            # store all training data
            all_store_data[epoch, 0] = epoch
            all_store_data[epoch, 1] = loss_cycle_B
            all_store_data[epoch, 2] = loss_D
            
            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(G_AB.state_dict(), './RESULTS/saved_models/%s/G_AB_%d.pth' % (stage, epoch))
                torch.save(G_BA.state_dict(), './RESULTS/saved_models/%s/G_BA_%d.pth' % (stage, epoch))
                torch.save(D_A.state_dict(), './RESULTS/saved_models/%s/D_A_%d.pth' % (stage, epoch))
                torch.save(D_B.state_dict(), './RESULTS/saved_models/%s/D_B_%d.pth' % (stage, epoch))
                
            
            if epoch % 2 == 0:
                VF.plot_output('A_' + str(epoch), num_epoch, real_A, fake_A, img_size, False, saveSVG=True)
                VF.plot_output('B_' + str(epoch), num_epoch, real_B, fake_B, img_size, False, saveSVG=True)
                
                
                
        # stop training time
        final_time = time.time()
        print('PROGRESS: Training Done!')        
        
        # get train_outputs and train_final_loss
        train_total_loss, train_outputs = VF.get_loss_and_output(G_AB, train_inputs, train_labels, train_params, 
                                                                 mini_batch_size, adversarial_loss, device)
        
        # get test_outputs and test_final_loss
        test_total_loss, test_outputs = VF.get_loss_and_output(G_AB, test_inputs, test_labels, test_params, 
                                                               mini_batch_size, adversarial_loss, device)
        print('PROGRESS: Outputs obtained!')
            

        
        # save output
        VF.save_data(train_outputs, 'train_outputs')
        VF.save_data(test_outputs, 'test_outputs')

        # plot results
        VF.result_plot(stage, all_store_data, adversarial_loss, True, saveSVG=True)
        VF.plot_output('input' + stage, num_epoch, train_labels, train_outputs, img_size, False, saveSVG=True)
        VF.plot_output(stage, num_epoch, test_labels, test_outputs, img_size, True, saveSVG=True)
        VF.plot_max_stress(test_labels, test_outputs, stage, all_store_data, adversarial_loss, True, saveSVG=True)
        VF.draw_stress_propagation_plot(test_labels[0:4], test_outputs[0:4], stage, all_store_data, adversarial_loss, True, saveSVG=True)
        VF.mean_and_CI(test_labels, test_outputs, stage, all_store_data, adversarial_loss, True, saveSVG=False)
           
        # save final model and state_dict
        VF.save_model(G_AB, stage)

        # obtain image difference
        img_diff = VF.image_difference(test_labels, test_outputs)

        # calculate max stress error
        overall_max_stress_error = VF.avg_max_stress_error(test_labels, test_outputs, img_size)

        time.sleep(5)

        # save all parameters
        model_specification = VF.net_specification(stage, str(train_inputs.size(0) + test_inputs.size(0)),
                                                   str(train_inputs.size(2)) +' x ' + str(train_inputs.size(3)),
                                                   str(test_inputs.size(2)) +' x ' + str(test_inputs.size(3)),
                                                   num_epoch, mini_batch_size, learning_rate, momentum, adversarial_loss,
                                                   optimizer_G, str(train_inputs.size(0)) + ' / ' + str(test_inputs.size(0)),
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
        VF.create_report(G_AB, stage, model_specification)
        VF.create_report_page2(G_AB, stage, model_specification)
        print('PROGRESS: Report Created!')



    # ============================== TESTING ======================================
    else:

        # read model
#        val_net = torch.load(validataion_model)
        val_net = G_AB
        val_net.state_dict = validation_model_state_dict
        test_output = val_net(test_inputs, test_params)
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

