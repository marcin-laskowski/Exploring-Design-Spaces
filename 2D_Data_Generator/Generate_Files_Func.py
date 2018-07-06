
# coding: utf-8

# In[73]:


import numpy as np
from decimal import Decimal
import os


# In[10]:


def generate_polygons(rmin, rmax, n_gons, n_out):
    """    this generates n_out polygons of n_gons sides
            from a circle with a radius between rmin and rmax """
    #Creating the vectors of radius (can be redundant)
    listr = np.random.ranf(n_out)*(rmax-rmin)+rmin

    #Initializing the Matrix of angles of size (n_gons, n_out)
    mat_theta = np.zeros((n_gons,n_out))
    thetanormal = [k*2*np.pi/n_gons for k in range(n_gons)]

    for i in range(n_out):
        mat_theta[:,i] = [np.random.normal(thetanormal[k], listr[i]/9) for k in range(n_gons)]

    x = listr*np.cos(mat_theta) #Xcoordinates
    y = listr*np.sin(mat_theta) #Ycoordinates

    return (x,y)


# In[60]:


def Generate_Inputs():
    # --------------------------  Compatibilities  -----------------------
    Node_Can_Have_Both_Fixed_And_Force = False  # Put True if you want to be able to put a force at a fixed node, 0 otherwise


                                #  1 for same
    Fixed_Dis_And_Rot_Same = 0  #  0 for don't care
                                # -1 for different

    # --------------------------  Parameters  -----------------------
    n_gons = 5    # Pentagon
    n_fixed_Displacement_nodes = 3
    n_fixed_Rotation_nodes = 0
    n_forces_at_nodes = 2
    n_fixed_rotation = 0
    n_fmomentum_at_nodes = 0
    n_pressure = 0


    # --------------------------  Errors and Warning Messages  -----------------------
    if not Node_Can_Have_Both_Fixed_And_Force and (n_forces_at_nodes + n_fixed_Displacement_nodes > n_gons):
        print('ERROR : Some forces will be applied on fixed nodes               please reduce the number of fixed nodes or nodes with forces so that their sum is below the number of nodes               or authorize to put force on fixed points')
        return 0

    if Node_Can_Have_Both_Fixed_And_Force and (n_forces_at_nodes + n_fixed_Displacement_nodes > n_gons):
        print('WARNING : Some forces will be applied on fixed nodes')


    # --------------------------  Fixed Nodes Displacement -----------------------
    #Choosing randomly the location of the fixed nodes for displacement
    ones = np.ones(n_fixed_Displacement_nodes)
    zeros = np.zeros(n_gons - n_fixed_Displacement_nodes)
    Ind_Fixed_Displacement_Nodes = np.concatenate((ones, zeros))
    np.random.shuffle(Ind_Fixed_Displacement_Nodes)

    # --------------------------  Fixed Nodes Rotation -----------------------
    #Choosing randomly the location of the fixed nodes for rotation
    if Fixed_Dis_And_Rot_Same == 1:
        Ind_Fixed_Rotation_Nodes = Ind_Fixed_Displacement_Nodes

    elif Fixed_Dis_And_Rot_Same == 0:
        ones = np.ones(n_fixed_Rotation_nodes)
        zeros = np.zeros(n_gons - n_fixed_Rotation_nodes)
        Ind_Fixed_Rotation_Nodes = np.concatenate((ones, zeros))
        np.random.shuffle(Ind_Fixed_Rotation_Nodes)

    elif Fixed_Dis_And_Rot_Same == -1:
        Ind_Fixed_Displacement_Rotation = np.zeros(n_gons)
        Ind_Available_For_Fixed_Rotation = np.argwhere(Ind_Fixed_Displacement_Nodes==0)







    # --------------------------  Forces on nodes  -----------------------
    #Choosing randomly the location of the forces

    if Node_Can_Have_Both_Fixed_And_Force:
        #Here We can put the Forces on any node since we allowed applying forces on fixed nodes
        ones = np.ones(n_forces_at_nodes)
        zeros = np.zeros(n_gons - n_forces_at_nodes)
        Ind_Forces = np.concatenate((ones, zeros))
        np.random.shuffle(Ind_Forces)

    else:
        #Here we can only put Forces on a non fixed node since we allowed applying forces on fixed nodes

        Ind_Forces = np.zeros(n_gons)
        Ind_Available_For_Forces = np.argwhere(Ind_Fixed_Displacement_Nodes==0)
        np.random.shuffle(Ind_Available_For_Forces)
        Ind_Forces[Ind_Available_For_Forces[0:n_forces_at_nodes]] = 1


    # --------------------------  Momentum on nodes  -----------------------
    #Here We can put the Forces on any node since we allowed applying forces on fixed nodes
    ones = np.ones(n_fmomentum_at_nodes)
    zeros = np.zeros(n_gons - n_fmomentum_at_nodes)
    Ind_Momentum = np.concatenate((ones, zeros))
    np.random.shuffle(Ind_Momentum)

    # --------------------------  Pressure  -----------------------
    n_pressure = 1 #np.int(1+2*np.random.rand())
    element = np.array([1+np.int(3*np.random.rand()) for i in range(n_pressure)])
    Value = np.array([np.int(30*np.random.rand())-15 for i in range(n_pressure)])
    side =  np.zeros(n_pressure)
    for i in range(n_pressure):
        if element[i] == 1:
            side[i] = np.int(2*np.random.rand())
        elif element[i] == 2:
            side[i] = np.int(0)
        elif element[i] == 3:
            side[i] = np.int(2*np.random.rand())





    # --------------------------  Fill the conditions  -----------------------
    F_X = Ind_Fixed_Displacement_Nodes
    F_Y = Ind_Fixed_Displacement_Nodes
    F_Z = np.zeros(5)  #If 2D

    FRx = Ind_Fixed_Rotation_Nodes
    FRy = Ind_Fixed_Rotation_Nodes
    FRz = np.zeros(5)  #If 2D

    FX = 10*np.random.rand(5)*Ind_Forces
    FY = 10*np.random.rand(5)*Ind_Forces
    FZ = np.zeros(5)  #If 2D

    MX = 10*np.random.rand(5)*Ind_Momentum
    MY = 10*np.random.rand(5)*Ind_Momentum
    MZ = np.zeros(5)  #If 2D



    return [F_X, F_Y, F_Z, FRx, FRy, FRz, FX, FY, FZ, MX, MY, MZ, Value, element, side]








# In[61]:


def Generate_File(filename,X,Y,Z,F_X, F_Y, F_Z, FRx, FRy, FRz, FX, FY, FZ, MX, MY, MZ, Value, element, side):
    file = open(filename,'w')
    file.write("TYPE \t VALUE \t ELEMENT \t SIDE\n")
    for i in range(element.size):
        file.write("pressure\t{}\t{}\t{}\n".format(int(Value[i]), int(element[i]), int(side[i])))


    file.write("X\t\t\tY\t\t\tZ\t\t\tF_X\tF_Y\tF_Z\tFRx\tFRy\tFRz\tFX\tFY\tFZ\tMX\tMY\tMZ\t\n")
    for i in range(5):
        file.write("{}{}".format('%.12e' % Decimal(X[i]),'\t'))
        file.write("{}{}".format('%.12e' % Decimal(Y[i]),'\t'))
        file.write("{}{}".format('%.12e' % Decimal(Z[i]),'\t'))

        file.write("{}{}".format(int(F_X[i]),'\t'))
        file.write("{}{}".format(int(F_Y[i]),'\t'))
        file.write("{}{}".format(int(F_Z[i]),'\t'))

        file.write("{}{}".format(int(FRx[i]),'\t'))
        file.write("{}{}".format(int(FRy[i]),'\t'))
        file.write("{}{}".format(int(FRz[i]),'\t'))

        file.write("{}{}".format('%.1e' % Decimal(FX[i]),'\t'))
        file.write("{}{}".format('%.1e' % Decimal(FY[i]),'\t'))
        file.write("{}{}".format('%.1e' % Decimal(FZ[i]),'\t'))

        file.write("{}{}".format('%.1e' % Decimal(MX[i]),'\t'))
        file.write("{}{}".format('%.1e' % Decimal(MY[i]),'\t'))
        file.write("{}{}".format('%.1e' % Decimal(MZ[i]),'\t'))

        file.write("{}".format('\n'))



# In[74]:


def Generate_n_Files(n):
    if not os.path.exists('./input_params'):
        os.mkdir('./input_params')
    X_TOT,Y_TOT = generate_polygons(1, 1, 5, n)
    Z_TOT = np.zeros((5,n))
    for i in range(n):
        X = X_TOT[:,i]
        Y = Y_TOT[:,i]
        Z = Z_TOT[:,i]
        [F_X, F_Y, F_Z, FRx, FRy, FRz, FX, FY, FZ, MX, MY, MZ, Value, element, side] = Generate_Inputs()
        filename = "./input_params/InputParams{}.txt".format(i)
        Generate_File(filename,X,Y,Z,F_X, F_Y, F_Z, FRx, FRy, FRz, FX, FY, FZ, MX, MY, MZ, Value, element, side)
        if i < n-1:
            del F_X, F_Y, F_Z, FRx, FRy, FRz, FX, FY, FZ, MX, MY, MZ, Value, element, side



def Generate_n_Files_same(n):
    if not os.path.exists('./input_params'):
        os.mkdir('./input_params')
    X_TOT,Y_TOT = generate_polygons(1, 1, 5, n)
    Z_TOT = np.zeros((5,n))
    [F_X, F_Y, F_Z, FRx, FRy, FRz, FX, FY, FZ, MX, MY, MZ, Value, element, side] = Generate_Inputs()
    for i in range(n):
        X = X_TOT[:,i]
        Y = Y_TOT[:,i]
        Z = Z_TOT[:,i]
        
        filename = "./input_params/InputParams{}.txt".format(i)
        Generate_File(filename,X,Y,Z,F_X, F_Y, F_Z, FRx, FRy, FRz, FX, FY, FZ, MX, MY, MZ, Value, element, side)




















