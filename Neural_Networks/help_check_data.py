import os
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from math import sqrt

import VisualizationFunction2 as VF

import warnings
warnings.filterwarnings("ignore")

# from IPython import get_ipython




# ========================= PARAMETERS ========================================
plot_step_images = False
save_PDF_all_data = True
create_best_data = False

device = torch.device('cpu')
img_size = 64


# =========================== PREPARE DATA ====================================
#Inputs = torch.from_numpy(np.load('./DATA/01_data_noPress/Images_DataSet.npy'))
#Labels = torch.from_numpy(np.load('./DATA/01_data_noPress/Labels_DataSet.npy'))
#Inputs = torch.from_numpy(np.load('./DATA/03_data_new/Images_DataSet.npy'))
#Labels = torch.from_numpy(np.load('./DATA/03_data_new/Labels_DataSet.npy'))
#Inputs = torch.from_numpy(np.load('./DATA/04_diffShapeBEST/Inputs_DataSet.npy'))
#Labels = torch.from_numpy(np.load('./DATA/04_diffShapeBEST/Labels_DataSet.npy'))
#Inputs = torch.from_numpy(np.load('./DATA/06_noPress_50force/Images_DataSet.npy'))
#Labels = torch.from_numpy(np.load('./DATA/06_noPress_50force/Labels_DataSet.npy'))
Labels = torch.from_numpy(np.load('./DATA/07_noPressNewShape/Labels_DataSet.npy'))
Inputs = torch.from_numpy(np.load('./DATA/07_noPressNewShape/Images_DataSet.npy'))

#Inputs = Variable(Inputs.float()).to(device)
#Labels = Variable(Labels.float()).to(device)

inputs = Inputs.view(Inputs.size(0), Inputs.size(3), Inputs.size(4))
labels = Labels.view(Labels.size(0), Labels.size(3), Labels.size(4))


# ===================== CALCULATE SUM OF THE LABELS ===========================
labels_sum = np.zeros((Labels.size(0), 2))
only_labels_sum = np.zeros((Labels.size(0), 1))
for i in range(Labels.size(0)):
    temp_img = labels[i, :, :]
    img = temp_img.numpy()
    sum_img = np.sum(img)
    labels_sum[i, 0] = i
    labels_sum[i, 1] = int(sum_img)
    only_labels_sum[i] = int(sum_img)
        
    
# =============================== VISUALIZE ===================================
labels_sum = np.array(labels_sum)
labels_sum_sorted = labels_sum[labels_sum[:,1].argsort()]

data = (labels_sum_sorted[:,1]).tolist()
index = list(range(Labels.size(0)))
#plt.xlabel('index of the image')
#plt.ylabel('Sum of one image (64x64)')
plt.title('Check Data')
plt.plot(index, data)
plt.savefig('plot.svg')


# =========================== PLOT IMAGES =====================================
# get indexes of the images
if plot_step_images == True:
    
    idx_0 = only_labels_sum.tolist().index([0.0])
    idx_1 = only_labels_sum.tolist().index([10003.0])
    idx_2 = only_labels_sum.tolist().index([20001.0])
    idx_3 = only_labels_sum.tolist().index([50025.0])
    idx_4 = only_labels_sum.tolist().index([245064.0])
    
    
    data_numbers = [idx_0, idx_1, idx_2, idx_3, idx_4]
    sum_of_img = [0, 10003, 20001, 50025, 245064]
    
    fig, ax = plt.subplots(figsize=(12,5), ncols=len(data_numbers), nrows=1)
    
    for i in range(len(data_numbers)):
        out = labels[data_numbers[i], :, :]
        temp_out = (Variable(out).data).cpu().numpy()
        ax[i].imshow(temp_out.T, extent=(0, img_size, 0, img_size), origin=sum_of_img[i])
    #    ax[i].set_xlabel('sum_{}'.format(sum_of_img[i]))
        ax[i].xaxis.set_label_text('sum_{}'.format(sum_of_img[i]))
    #    ax[i].axis('off')
        ax[i].set_title('label idx_{}'.format(data_numbers[i]))
        
        
    
    # adjust image
    w = int(26 / 2.5)  # 30
    h = int(12 / 2.5)  # 10
    fig.set_size_inches(w,h)
    
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(top = 0.8)
    
    fig.savefig('check_labels.svg')


# ===================== PLOT MATRIX WITH LABELS ===============================
if save_PDF_all_data == True:

    data_numbers = labels_sum_sorted[:,0]
    data_numbers = data_numbers.astype(np.int64)
    sum_of_img = labels_sum_sorted[:,1]
    sum_of_img = sum_of_img.astype(np.int64)
    
    ncols = 10
    nrows = 10
    n = 0
    
    for batch in range(100):
    
    #    data_numbers = data_numbers[batch:batch+(ncols*nrows)]
    #    sum_of_img = sum_of_img[batch:batch+(ncols*nrows)]
    
        fig, ax = plt.subplots(figsize=(12,5), ncols=ncols, nrows=nrows)
        
        for i in range (nrows):
            for j in range(ncols):
                out = labels[data_numbers[n], :, :]
                temp_out = (Variable(out).data).cpu().numpy()
                ax[i][j].imshow(temp_out.T, extent=(0, img_size, 0, img_size), origin='1')
                ax[i][j].set_axis_off()
                ax[i][j].set_title('label idx_{}'.format(data_numbers[n]))
                ax[i][j].set_xlabel(str(sum_of_img[n]))
                n += 1
            
        
        # adjust image
        w = int(100 / 2.5)  # 30
        h = int(100 / 2.5)  # 10
        fig.set_size_inches(w,h)
        
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.subplots_adjust(top = 0.8)
        
        if not os.path.exists('./LABELS_7'):
            os.mkdir('./LABELS_7')
        
        fig.savefig('./LABELS_7/all_labels_' + str(batch) + '.pdf')
        
        print(str(batch) + '%')
    print('DONE!')



# ========================== CREATE NEW DATA ==================================
if create_best_data == True:
    
    new_Inputs = np.load('./DATA/01_data_noPress/Images_DataSet.npy')
    new_Labels = np.load('./DATA/01_data_noPress/Labels_DataSet.npy')
    new_Params = np.load('./DATA/01_data_noPress/InputParams_matrix.npy')
    new_FixLoad = np.load('./DATA/01_data_noPress/Input_fix_and_force_matrix.npy')
    
    new_indexes = labels_sum_sorted[1000:, 0]
    new_indexes = new_indexes.astype(np.int64)
    
    Inputs2 = np.zeros((9000, 1, 1, 64, 64))
    Labels2 = np.zeros((9000, 1, 1, 64, 64))
    Params2 = np.zeros((9000, 5, 6))
    FixLoad2 = np.zeros((9000, 5, 4))
    
    test = np.zeros((9000, 1))
    
    n = 0
    for i in range(10000):
        if i in new_indexes:
            Inputs2[n,:,:,:,:] = new_Inputs[i, :, :, :, :]
            Labels2[n,:,:,:,:] = new_Labels[i, :, :, :, :]
            Params2[n,:,:] = new_Params[i, :, :]
            FixLoad2[n,:,:] = new_FixLoad[i, :, :]
    #        test[n,0] = i
            n += 1
        else:
            pass
        
    
    ImageFile = './DATA/05_noPressBEST/Inputs_DataSet'
    LabelFile = './DATA/05_noPressBEST/Labels_DataSet'
    ParamFile = './DATA/05_noPressBEST/Params_DataSet'
    FixLoadFile = './DATA/05_noPressBEST/FixLoads_DataSet'
    
    np.save(ImageFile, Inputs2)
    np.save(LabelFile, Labels2)
    np.save(ParamFile, Params2)
    np.save(FixLoadFile, FixLoad2)

