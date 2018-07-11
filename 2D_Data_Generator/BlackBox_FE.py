'''
MARC Software: Code to generate the data for the project:
EXPLORING DESIGN SPACES - STAGE 1
'''


import os
from py_post import *
import sys
import os
import math
import numpy as np

if not os.path.exists('./input_params'):
    os.mkdir('./input_params')


def BlackBox_FE(file_name, index_n):

    # ================================== CREATE FOLDERS ===========================================

    # vonMises
    if not os.path.exists('./vonMises'):
        os.mkdir('./vonMises')

    # ================================= READ INPUT_PARAMS =====================================

    file_name = './input_params/{}'.format(file_name)
    file = open(file_name, "r")
    lines = list(file)
    file.close()

    # ================================= MODEL GENERATION ======================================

    file = open("original.proc", "r")
    proc = list(file)
    newproc = []
    newproc.extend(proc[0:1])
    file.close()
    kk = 1
    kj = 1
    firstline = lines[kj].split('\t')
    pressure = []
    elepress = []
    edgepress = []

    if firstline[0] == "pressure":
        while (firstline[0] == "pressure"):
            kk = kj+2
            kj = kj+1
            pressure.extend([firstline[1]])
            elepress.extend([firstline[2]])
            edgepress.extend([firstline[3]])
            firstline = lines[kj].split('\t')

        coord = np.zeros(shape=(len(lines)-kk, 3))
        fixation = np.zeros(shape=(len(lines)-kk, 6))
        load = np.zeros(shape=(len(lines)-kk, 6))

    else:
        kk += 1

        coord = np.zeros(shape=(len(lines)-kk, 3))
        fixation = np.zeros(shape=(len(lines)-kk, 6))
        load = np.zeros(shape=(len(lines)-kk, 6))

    kj = 0
    for j in range(kk, len(lines)):
        indat = lines[j].split('\t')
        coord[kj][0] = indat[0]
        coord[kj][1] = indat[1]
        coord[kj][2] = indat[2]
        fixation[kj][0] = indat[3]
        fixation[kj][1] = indat[4]
        fixation[kj][2] = indat[5]
        fixation[kj][3] = indat[6]
        fixation[kj][4] = indat[7]
        fixation[kj][5] = indat[8]
        load[kj][0] = indat[9]
        load[kj][1] = indat[10]
        load[kj][2] = indat[11]
        load[kj][3] = indat[12]
        load[kj][4] = indat[13]
        load[kj][5] = indat[14]
        kj = kj+1

    for j in range(0, len(coord)):
        newproc.append('%s\t%s\t%s\n' % (coord[j][0], coord[j][1], coord[j][2]))

    newproc.extend(proc[1:6])
    if(kk > 1):
        for j in range(0, len(pressure)):
            newproc.append('*new_apply *apply_type edge_load\n')
            newproc.append('%s%s\n' % ('*apply_name Pressure', (j+1)))
            newproc.append('%s%s%s\n' % ('*apply_dof p *apply_dof_value p ', pressure[j], ' #'))
            statement = '%s:%s' % (elepress[j], edgepress[j])
            newproc.append('%s%s%s\n' % ('*add_apply_edges ', statement.strip('\n'), ' #'))

    kk = 0
    for j in range(0, len(fixation)):
        summ = np.sum(fixation[j])
        if summ >= 1:
            kk = kk+1
            newproc.append('*new_apply *apply_type fixed_displacement\n')
            newproc.append('%s%s\n' % ('*apply_name Fixation', (kk)))
            tag = ['x', 'y', 'z', 'rx', 'ry', 'rz']
            for i in range(0, 5):
                if fixation[j][i] == 1:
                    newproc.append('%s%s\n' % ('*apply_dof ', tag[i]))
            newproc.append('%s%s%s\n' % ('*add_apply_nodes ', (j+1), ' #'))
    # newproc.extend(proc[6:])

    kk = 0
    for j in range(0, len(load)):
        summ = np.sum(load[j])
        if summ > 0:
            kk = kk+1
            newproc.append('*new_apply *apply_type point_load\n')
            newproc.append('%s%s\n' % ('*apply_name Load', (kk)))
            tag = ['x', 'y', 'z', 'rx', 'ry', 'rz']
            for i in range(0, 5):
                if load[j][i] != 0:
                    newproc.append('%s%s%s%s%s%s\n' % ('*apply_dof ', tag[i], ' *apply_dof_value ',
                                                       tag[i], ' ', load[j][i]))
            newproc.append('%s%s%s\n' % ('*add_apply_nodes ', (j+1), ' #'))
    newproc.extend(proc[6:])

    # ================================= SAVE BASELINE =========================================
    file = open('baseline.proc', "w")
    file.writelines(newproc)
    file.close()
#    ATTENTION HERE
    newproc.extend(proc[1:20])

    # ================================ EXECUTE MODEL GENERATION ===============================
    os.system("ModelGenerator.bat")

    # =================================== MODEL SIMULATION ====================================
    os.system("ModelRun.bat")

    # ================================== RESULTS EXTRAPOLATION ================================
    p = post_open("load1_load1.t16")
    nincs = p.increments()
    p.extrapolation("linear")
    p.moveto(nincs-1)
    nelements = p.elements()
    dat = p.element_scalar(1, 0)
    SC = [[0 for y in range(4)] for x in range(nelements)]

    for j in range(0, nelements):
        dat = p.element_scalar(j, 0)
        nod1 = p.node(dat[0].id-1)
        nod2 = p.node(dat[1].id-1)
        nod3 = p.node(dat[2].id-1)
        SC[j][0] = (nod1.x+nod2.x+nod3.x)/3
        SC[j][1] = (nod1.y+nod2.y+nod3.y)/3
        SC[j][2] = (nod1.z+nod2.z+nod3.z)/3
        SC[j][3] = dat[0].value
    p.close()

# ================================= READ VONMISES =========================================
    #first_row = ('X Y Z VonMises')
    file = open('./vonMises/VonMisesCentroid{}.txt'.format(index_n), 'w')
    np.savetxt(file, SC)  # ), header=first_row)
    file.close()


# ================================= REMOVE FILES =========================================
    os.remove('baseline.proc')
    os.remove('load1.mud')
    os.remove('load1_load1.dat')
    os.remove('load1_load1.out')
    os.remove('load1_load1.sts')
    os.remove('load1_load1.t16')
    os.remove('mentat.log')
    os.remove('mentat.proc')
