"""
EXPLORING DESIGN SPACES - Main File

STAGE 0, STAGE 1.1, STAGE 1.2, STAGE 1.3
"""

import torch
import argparse

import stage_0 as STAGE_0
import ModelStage_0 as M0
import stage_1_2 as STAGE_1_2
import ModelStage_1_2 as M12
import stage_1_3 as STAGE_1_3
import ModelStage_1_3 as M13
import stage_1_4 as STAGE_1_4
import ModelStage_1_4 as M14
import stage_1_5 as STAGE_1_5
import ModelStage_1_5 as M15
import stage_1_6 as STAGE_1_6
import ModelStage_1_6 as M16


# ====================== PARAMETERS TO INVESTIGATE ============================
parser = argparse.ArgumentParser(description='Stage_1_3')
parser.add_argument('--name', type=str, default='LinConv_1',
                    help='Name of the neural network architecture')
parser.add_argument('--device_type', type=str, default='cpu',
                    help='perform calculations on CPU or GPU')
parser.add_argument('--batch', type=int, default=40, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--split', type=float, default=7/10, metavar='M',
                    help='training / testing split (default: 7/10)')
parser.add_argument('--dataset', type=str, default='05_noPressBEST', metavar='M',
                    help='choose dataset for the model (default: 05_noPressBEST)')

args = parser.parse_args()


args.device_type = 'cuda'
device = torch.device(args.device_type)
args.epochs = 2000
args.batch = 40


# Dataset
args.dataset = '10_diffShapesBEST'
args.split = 7/8
args.lr = 0.0001

# ============== NEURAL NETWORKS ARCHITECTURES TO INVESTIGATE =================

# STAGE 0.0
args.name = 'LinLin_0'
print(' ######### Stage 0 ' + args.name + ' ######### ')
STAGE_0.network(args, M0.LinLin_0())

args.name = 'LinLin_1'
print(' ######### Stage 0 ' + args.name + ' ######### ')
STAGE_0.network(args, M0.LinLin_1())

args.name = 'ConvConv_0'
print(' ######### Stage 0 ' + args.name + ' ######### ')
STAGE_0.network(args, M0.ConvConv_0())

args.name = 'ConvConv_1'
print(' ######### Stage 0 ' + args.name + ' ######### ')
STAGE_0.network(args, M0.ConvConv_1())

args.name = 'ConvLinLinConv_0'
print(' ######### Stage 0 ' + args.name + ' ######### ')
STAGE_0.network(args, M0.ConvLinLinConv_0())

args.name = 'ConvLinLinConv_1'
print(' ######### Stage 0 ' + args.name + ' ######### ')
STAGE_0.network(args, M0.ConvLinLinConv_1())



# STAGE 1.3
args.name = 'LinConv_0'
print(' ######### Stage 1.3 ' + args.name + ' ######### ')
STAGE_1_3.network(args, M13.LinConv_0())
#
args.name = 'LinConv_1'
print(' ######### Stage 1.3 ' + args.name + ' ######### ')
STAGE_1_3.network(args, M13.LinConv_1())

args.name = 'LinConv_2'
print(' ######### Stage 1.3 ' + args.name + ' ######### ')
STAGE_1_3.network(args, M13.LinConv_2())

args.name = 'LinConv_3'
print(' ######### Stage 1.3 ' + args.name + ' ######### ')
STAGE_1_3.network(args, M13.LinConv_3())

args.name = 'LinConv_4'
print(' ######### Stage 1.3 ' + args.name + ' ######### ')
STAGE_1_3.network(args, M13.LinConv_4())

args.name = 'LinConv_5'
print(' ######### Stage 1.3 ' + args.name + ' ######### ')
STAGE_1_3.network(args, M13.LinConv_5())

args.name = 'LinConv_7'
print(' ######### Stage 1.3 ' + args.name + ' ######### ')
STAGE_1_3.network(args, M13.LinConv_7())


# STAGE 1.2
args.name = 'LinLinConv_0'
print(' ######### Stage 1.2 ' + args.name + ' ######### ')
STAGE_1_2.network(args, M12.LinLinConv_0())

args.name = 'LinLinConv_1'
print(' ######### Stage 1.2 ' + args.name + ' ######### ')
STAGE_1_2.network(args, M12.LinLinConv_1())

args.name = 'ConvLinConv_0'
print(' ######### Stage 1.2 ' + args.name + ' ######### ')
STAGE_1_2.network(args, M12.ConvLinConv_0())

args.name = 'ConvLinConv_1'
print(' ######### Stage 1.2 ' + args.name + ' ######### ')
STAGE_1_2.network(args, M12.ConvLinConv_1())
#
args.name = 'ConvLinConv_2'
print(' ######### Stage 1.2 ' + args.name + ' ######### ')
STAGE_1_2.network(args, M12.ConvLinConv_2())

args.name = 'ConvLinConv_3'
print(' ######### Stage 1.2 ' + args.name + ' ######### ')
STAGE_1_2.network(args, M12.ConvLinConv_3())

args.name = 'ConvLinConv_4'
print(' ######### Stage 1.2 ' + args.name + ' ######### ')
STAGE_1_2.network(args, M12.ConvLinConv_4())




# STAGE 1.4   
args.name = 'ConvLinConv_0'
print(' ######### Stage 1.4 ' + args.name + ' ######### ')
STAGE_1_4.network(args, M14.ConvLinConv_0())

args.name = 'ConvLinConv_1'
print(' ######### Stage 1.4 ' + args.name + ' ######### ')
STAGE_1_4.network(args, M14.ConvLinConv_1())

args.name = 'ConvLinConv_2'
print(' ######### Stage 1.4 ' + args.name + ' ######### ')
STAGE_1_4.network(args, M14.ConvLinConv_2())

args.name = 'ConvLinConv_3'
print(' ######### Stage 1.4 ' + args.name + ' ######### ')
STAGE_1_4.network(args, M14.ConvLinConv_3())

args.name = 'ConvLinConv_6'
print(' ######### Stage 1.4 ' + args.name + ' ######### ')
STAGE_1_4.network(args, M14.ConvLinConv_6())


# STAGE 1.5
args.name = 'LinConv_0'
print(' ######### Stage 1.5 ' + args.name + ' ######### ')
STAGE_1_5.network(args, M15.LinConv_0())

args.name = 'LinConv_1'
print(' ######### Stage 1.5 ' + args.name + ' ######### ')
STAGE_1_5.network(args, M15.LinConv_1())

args.name = 'LinConv_2'
print(' ######### Stage 1.5 ' + args.name + ' ######### ')
STAGE_1_5.network(args, M15.LinConv_2())

args.name = 'LinConv_3'
print(' ######### Stage 1.5 ' + args.name + ' ######### ')
STAGE_1_5.network(args, M15.LinConv_3())

args.name = 'LinConv_4'
print(' ######### Stage 1.5 ' + args.name + ' ######### ')
STAGE_1_5.network(args, M15.LinConv_4())

args.name = 'LinConv_5'
print(' ######### Stage 1.5 ' + args.name + ' ######### ')
STAGE_1_5.network(args, M15.LinConv_5())

args.name = 'LinConv_6'
print(' ######### Stage 1.5 ' + args.name + ' ######### ')
STAGE_1_5.network(args, M15.LinConv_6())

args.name = 'LinConv_7'
print(' ######### Stage 1.5 ' + args.name + ' ######### ')
STAGE_1_5.network(args, M15.LinConv_7())

args.name = 'LinConv_8'
print(' ######### Stage 1.5 ' + args.name + ' ######### ')
STAGE_1_5.network(args, M15.LinConv_8())


# STAGE 1.6
#args.name = 'GAN_Generator_Discriminator'
#print(' ######### Stage 1.6 ' + args.name + ' ######### ')
#STAGE_1_6.network(args, M16.Generator_AB(), M16.Generator_BA(), M16.Discriminator())

