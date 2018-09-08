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

import scipy.io as sio  # to save mat file
import scipy.misc  # o save .jpg
from PIL import Image  # to invert image
import PIL.ImageOps  # to invert image
from scipy.interpolate import griddata

import GenerateDataFunctions as GDF
import VisualizationFunction2 as VF

import warnings
warnings.filterwarnings("ignore")
start_time = time.time()

from IPython import get_ipython




# ========================= PARAMETERS ========================================
read_input_data = True
get_params_and_fixforce = False
add_additional_channels = False
standard_vis = False
separate_images = True

device_type = 'cpu'
learning_rate = 0.001       # 0.001
mini_batch_size = 100       # 100
momentum = 0.9
img_size = 64
num_channels = 1

split = 4/5                 # 6/10, 5000


device = torch.device(device_type)  # cpu or cuda

validation_model = '.pt'


#dataset = '05_noPressBEST'
#dataset = '08_noPressNewShape2'
#dataset = '04_diffShapeBEST'
#dataset = '11_dataset'
dataset = '10_diffShapesBEST'


# =========================== PREPARE DATA ====================================
if get_params_and_fixforce == True:
    
    # NOTE!
    # change in 'automate_get_params' and 'automate_get_fix_and_force' paths
    # to folder with data
    
    number_of_elements = 10000
    
    # get parameters
    points_and_fix = GDF.automate_get_params(number_of_elements)
    
    input_data = torch.zeros((number_of_elements, 5, 6))
    input_data = input_data.float()
    for i in range(number_of_elements):
        input_data[i] = torch.from_numpy(points_and_fix[i])
    
    input_data = input_data.cpu().numpy()
    np.save('Params_DataSet', input_data)
    
    # get force and fixation
    force_and_fix = GDF.automate_get_fix_and_force(number_of_elements)
    
    input_data = torch.zeros((number_of_elements, 5, 4))
    input_data = input_data.float()
    for i in range(number_of_elements):
        input_data[i] = torch.from_numpy(force_and_fix[i])
    
    input_data = input_data.cpu().numpy()
    np.save('FixLoads_DataSet', input_data)
    
    
    
if add_additional_channels == True:
    
    Labels = np.load('./DATA/' + dataset + '/Labels_DataSet.npy')
    Inputs = np.load('./DATA/' + dataset + '/Inputs_DataSet.npy')

    new_Inputs, new_Labels = VF.X_and_Y_channels(Inputs, Labels, img_size)
    
    np.save('Inputs_3', new_Inputs)
    np.save('Labels_3', new_Labels)




# ==================== INPUT DATA =============================================
if read_input_data == True:
    
#    Inputs = input_data.view(input_data.size(0), 1, 1, 5, 6)

    Labels = torch.from_numpy(np.load('./DATA/' + dataset + '/Labels_DataSet.npy'))
    Inputs = torch.from_numpy(np.load('./DATA/' + dataset + '/Inputs_DataSet.npy'))
#    Params = torch.from_numpy(np.load('./DATA/' + dataset + '/Params_DataSet.npy'))
#    FixLoads = torch.from_numpy(np.load('./DATA/' + dataset + '/FixLoads_DataSet.npy'))
#    
#    Params = Params.view(Params.size(0), 1, 1, 5, 6)
#    FixLoads = FixLoads.view(FixLoads.size(0), 1, 1, 5, 4)

#    Labels = torch.from_numpy(np.load('./DATA/' + dataset + '/train_outputs.npy'))
#    Inputs = torch.from_numpy(np.load('./DATA/' + dataset + '/test_outputs.npy'))
    
    #Inputs = Inputs[:8000]
    #Labels = Labels[:8000]
    #train_inputs, train_labels, test_inputs, test_labels = VF.load_data(Inputs, Labels, split, mini_batch_size, device, img_size)
    #train_params, test_params = VF.load_params(Params, split, mini_batch_size, device, img_size)
    #train_fixloads, test_fixloads = VF.load_params(FixLoads, split, mini_batch_size, device, img_size)




# ========================= VISUALIZE =========================================
if standard_vis == True:
    
    train_inputs = Inputs
    train_labels = Inputs
#    train_params = Params
#    train_fixloads = FixLoads
    
    
    vis_out = train_labels.view(train_labels.size(0) * train_labels.size(1), 64, 64)
    vis_inputs = train_inputs.view(train_inputs.size(0) * train_inputs.size(1), 64, 64)
    
    num_1 = 0
    num_2 = 1
    num_3 = 2
    num_4 = 3
    num_5 = 4
    num_6 = 5
    num_7 = 6
    num_8 = 7
    
    img_0 = vis_inputs[num_1, :, :]
    img_1 = vis_out[num_5, :, :]
    img_2 = vis_inputs[num_2, :, :]
    img_3 = vis_out[num_6, :, :]
    img_4 = vis_inputs[num_3, :, :]
    img_5 = vis_out[num_7, :, :]
    img_6 = vis_inputs[num_4, :, :]
    img_7 = vis_out[num_8, :, :]
    
    grid_z0 = (Variable(img_0).data).cpu().numpy()
    grid_z1 = (Variable(img_1).data).cpu().numpy()
    grid_z2 = (Variable(img_2).data).cpu().numpy()
    grid_z3 = (Variable(img_3).data).cpu().numpy()
    grid_z4 = (Variable(img_4).data).cpu().numpy()
    grid_z5 = (Variable(img_5).data).cpu().numpy()
    grid_z6 = (Variable(img_6).data).cpu().numpy()
    grid_z7 = (Variable(img_7).data).cpu().numpy()
    
    plt.subplot(241)
    plt.imshow(grid_z0.T, extent=(0, 64, 0, 64), origin='1')
    plt.subplot(242)
    plt.imshow(grid_z1.T, extent=(0, 64, 0, 64), origin='2')
    plt.subplot(243)
    plt.imshow(grid_z2.T, extent=(0, 64, 0, 64), origin='3')
    plt.subplot(244)
    plt.imshow(grid_z3.T, extent=(0, 64, 0, 64), origin='4')
    plt.subplot(245)
    plt.imshow(grid_z4.T, extent=(0, 64, 0, 64), origin='1')
    plt.subplot(246)
    plt.imshow(grid_z5.T, extent=(0, 64, 0, 64), origin='2')
    plt.subplot(247)
    plt.imshow(grid_z6.T, extent=(0, 64, 0, 64), origin='3')
    plt.subplot(248)
    plt.imshow(grid_z7.T, extent=(0, 64, 0, 64), origin='4')
    
    #plt.show()
    plt.savefig('./images.pdf')
    
    
    
    """
    print(train_params[num_1,0,:,:])
    #print(train_fixloads[num_1,0,0,:,:])
    print(train_params[num_2,0,:,:])
    #print(train_fixloads[num_2,0,0,:,:])
    print(train_params[num_3,0,:,:])
    #print(train_fixloads[num_3,0,0,:,:])
    print(train_params[num_4,0,:,:])
    #print(train_fixloads[num_4,0,0,:,:])
    
    #temp = train_params[4,0,0,:,0:2]
    #temp2 = temp.cpu().numpy()
    #plt.plot(temp2[:,0], temp2[:,1])
    #plt.savefig('test.svg')
    """

if separate_images == True:
    
    train_inputs = Inputs
    train_labels = Labels   
    vis_inputs = train_inputs.view(train_inputs.size(0) * train_inputs.size(1), 64, 64)
    vis_out = train_labels.view(train_labels.size(0) * train_labels.size(1), 64, 64)
    
    for i in range(0,14):
        
        image = np.squeeze(train_inputs[i].cpu().numpy())
        
        # create empty folder
        if not os.path.exists('./images'):
            os.mkdir('./images')
    
        # create empty folder
        if not os.path.exists('./images/polygons'):
            os.mkdir('./images/polygons')
    
        image_temp = np.multiply(image, 255)
        scipy.misc.imsave('./images/polygons/{}_pentagon{}.png'.format(dataset, i), image_temp)
    
        image_inv = Image.open('./images/polygons/{}_pentagon{}.png'.format(dataset, i))
        inverted_image = PIL.ImageOps.invert(image_inv)
        inverted_image.save('./images/polygons/{}_inv_pentagon{}.png'.format(dataset, i))

