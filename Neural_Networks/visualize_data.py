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
learning_rate = 0.001       # 0.001
num_epoch = 200
mini_batch_size = 200       # 100
momentum = 0.9
img_size = 64
num_channels = 1

split = 2/4                 # 6/10, 5000


device = torch.device(device_type)  # cpu or cuda

validation_model = '.pt'


# =========================== PREPARE DATA ====================================

number_of_elements = 10000
points_and_fix = GDF.automate_get_params(number_of_elements)

input_data = torch.zeros((number_of_elements, 5, 6))
input_data = input_data.float()
for i in range(number_of_elements):
    input_data[i] = torch.from_numpy(points_and_fix[i])


# ==================== INPUT DATA =============================================
Inputs = input_data.view(input_data.size(0), 1, 1, 5, 6)
#Labels = np.load('./DATA/02_data_diffShape/Labels_DataSet.npy')
#Labels = np.load('./DATA/01_data_noPress/Labels_DataSet.npy')
Labels = np.load('./DATA/03_data_new/Labels_DataSet.npy')
Inputs2 = np.load('./DATA/03_data_new/Images_DataSet.npy')

Inputs = Inputs[:8000]
Labels = Labels[:8000]
train_inputs, train_labels, test_inputs, test_labels = VF.load_data(Inputs2, Labels, split, mini_batch_size, device, img_size)





# ========================= VISUALIZE =========================================

vis_inputs = train_out.view(train_out.size(0) * train_out.size(1), 64, 64)
vis_out = train_label.view(train_label.size(0) * train_label.size(1), 64, 64)

num_1 = 1
num_2 = 2

img_0 = vis_inputs[num_1, :, :]
img_1 = vis_out[num_1, :, :]
img_2 = vis_inputs[num_2, :, :]
img_3 = vis_out[num_2, :, :]

grid_z0 = (Variable(img_0).data).cpu().numpy()
grid_z1 = (Variable(img_1).data).cpu().numpy()
grid_z2 = (Variable(img_2).data).cpu().numpy()
grid_z3 = (Variable(img_3).data).cpu().numpy()

plt.subplot(221)
plt.imshow(grid_z0.T, extent=(0, 64, 0, 64), origin='1')
#    plt.title('1')
plt.subplot(222)
plt.imshow(grid_z1.T, extent=(0, 64, 0, 64), origin='2')
#    plt.title('2')
plt.subplot(223)
plt.imshow(grid_z2.T, extent=(0, 64, 0, 64), origin='3')
#    plt.title('3')
plt.subplot(224)
plt.imshow(grid_z3.T, extent=(0, 64, 0, 64), origin='4')
#    plt.title('4')

plt.show(hold=False)
#plt.savefig('./DATA/images.pdf')


