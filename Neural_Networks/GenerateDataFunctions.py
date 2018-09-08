"""
points_to_image - func that convert InpuParams.txt to matrix with values 0 and 1
stress_to_image - func that create matrix with values from 0 to 1 based on vonMises
automate_points_stress - create two 3D matrices with polygon and stress values
get_params - create matrix with the input parameters based on InputParams.txt
automate_get_params - create 3D matrix with all input parameters
get_pressure - get the pressure from the InputParams.txt
automate_get_pressure - create 3D matrix with all input pressures.
save_as_img - save matrix as an image

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  # to save mat file
import scipy.misc  # o save .jpg
from PIL import Image  # to invert image
import PIL.ImageOps  # to invert image
from scipy.interpolate import griddata


###################################################################################
###################################################################################

def points_to_image(file_name, input_poly, img_size, save_png=False, image_num=0):
    """
    FUNCTION THAT CREATES POINTS TO MATRIX WITH 0 AND 1

    file_name: polygon coordinates from the file: 'input_params.txt'
    input_poly: type of polygon (eg. input_polygon = 5, when we have pentagon)
    img_size: intiger value (eg. img_size = 128)
    save_jpg: it True it saves the picture
    image_num: number of the image
    ---------------------------------------------
    example:
    file_name = 'InputParams.txt'
    input_poly = 5
    img_size = 128
    image_num = 1
    """

    # =========================== INPUT DATA ======================================

    # open and read file
    file = open(file_name, 'r')
    lines = list(file)
    file.close()

    x = np.zeros((input_poly, 1))
    y = np.zeros((input_poly, 1))

    # get the coordinats of the polygon from the .txt file
    for i in range(len(lines)):
        split_line = lines[i].split('\t')
        if (split_line[0] == 'X'):
            k = i + 1
            pos = 0
            for m in range(k, k+input_poly):
                split_line = lines[m].split('\t')
                x[pos, 0] = float(split_line[0])
                y[pos, 0] = float(split_line[1])
                pos += 1

#    xy = np.concatenate((x, y), axis=1)

    # =========================== RESCALE =========================================

    # matrix rescaling
    x2 = (np.multiply(x, img_size/2)) + (img_size/2)
    y2 = (np.multiply(y, img_size/2)) + (img_size/2)
    xy2 = np.concatenate((x2, y2), axis=1)

    # ============================ FILL FUNCTION ==================================

#    pentagon = plt.fill(x2, y2, 'k')
#
#    plt.axes().set_aspect('equal', 'datalim')
#    plt.savefig('pentagon.jpg', dpi=300)
#    plt.show()

    # ===================== CHECK POINTS INSIDE POLYGON ===========================

    def point_in_poly(x, y, poly, img_size):

        # check if point is in the polygon nodes
        if not y == img_size/2:
            if (x, y) in poly:
                return 1

        # check if point is on a boundary
        for i in range(len(poly)):
            p1 = None
            p2 = None
            if i == 0:
                p1 = poly[0]
                p2 = poly[1]
            else:
                p1 = poly[i-1]
                p2 = poly[i]
            if p1[1] == p2[1] and p1[1] == y and x > min(p1[0], p2[0]) and x < max(p1[0], p2[0]):
                return 1

        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(n+1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y

        if inside:
            return 1
        else:
            return 0

    # create matrix with zeros
    image = np.zeros((img_size, img_size))
#    print(image)
#    print(len(image))

    # ========================= CREATE 0 / 1 MATRIX ===============================
    for vertical in range(img_size):
        for horizontal in range(img_size):
            temp = point_in_poly(horizontal, vertical, xy2, img_size)
            if temp == 1:
                image[vertical, horizontal] = 1

    # ========================= SAVE AS .MAT FILE =================================

    # save .mat file
#    sio.savemat('image.mat', {'image':image})

    # ========================== SAVE AS IMAGE ====================================

    # save .png
    if save_png == True:

        # create empty folder
        if not os.path.exists('./images'):
            os.mkdir('./images')

        # create empty folder
        if not os.path.exists('./images/polygons'):
            os.mkdir('./images/polygons')

        image_temp = np.multiply(image, 255)
        scipy.misc.imsave('./images/polygons/pentagon{}.png'.format(image_num), image_temp)

        image_inv = Image.open('./images/polygons/pentagon{}.png'.format(image_num))
        inverted_image = PIL.ImageOps.invert(image_inv)
        inverted_image.save('./images/polygons/inv_pentagon{}.png'.format(image_num))

    return image


###################################################################################
###################################################################################

def stress_to_image(stress_name, pentagon_matrix, img_size, save_png=False, image_num=0):
    """
    stress_name: .txt file with X, Y, Z and VonMises Stresses
    pentagon_name: .txt with the input values of the pentagon
    img_size: intiger value (eg. img_size = 128)
    output: matrix (img_size x img_size) with stresses
    save_jpg: it True it saves the picture
    ---------------------------------------------
    example:
    stress_name = 'VonMises@Centroid.txt'
    pentagon_name = 'InputParams.txt'
    img_size = 128
    save_jpg = False
    img_num = 1
    """

    # =========================== INPUT DATA ======================================

    # open and read file
    file = open(stress_name, 'r')
    lines = list(file)
    file.close()

    stress = np.zeros((len(lines), 3))

    # get the coordinates and VonMises stresses
    for i in range(len(lines)):
        split_line = lines[i].split(' ')
        stress[i, 0] = float(split_line[0])  # x-coordinate
        stress[i, 1] = float(split_line[1])  # y-coordinate
        stress[i, 2] = float(split_line[3])  # vonMises stress values

#    print(stress)

    # =========================== RESCALE COORDINATES ==============================

    # copy stress matrix for rescaling
    stress2 = stress.copy()

    # matrix rescaling (but only in terms of x and y coordinates)
    # we are not rescaling vonMises stresses
    stress2[:, 0] = (np.multiply(stress2[:, 0], img_size/2)) + (img_size/2)
    stress2[:, 1] = (np.multiply(stress2[:, 1], img_size/2)) + (img_size/2)

    # ============================ RESCALE STRESSES ================================

    res_mat = stress2.copy()
    #res_mat[:, 0:2] = np.round(res_mat[:, 0:2])

    max_val = np.amax(res_mat[:, 2])
    min_val = np.amin(res_mat[:, 2])

    # Normalization
    if min_val <= 0:
        res_mat[:, 2] = np.add(res_mat[:, 2], min_val)
    elif min_val == 0:
        pass
    else:
        res_mat[:, 2] = np.subtract(res_mat[:, 2], min_val)

    res_mat[:, 2] = np.divide(res_mat[:, 2], max_val)

    # ========================= INTERPOLATE GRID DATA ==============================

    # create grid of the size: img_size x img_size
    grid_x, grid_y = np.mgrid[0:img_size, 0:img_size]

    # interpolate the data
    grid_z0 = griddata(res_mat[:, 0:2], res_mat[:, 2], (grid_y, grid_x), method='nearest')
    grid_z1 = griddata(res_mat[:, 0:2], res_mat[:, 2], (grid_y, grid_x), method='linear')
    grid_z2 = griddata(res_mat[:, 0:2], res_mat[:, 2], (grid_y, grid_x), method='cubic')

    # obtain stress with the same shape as pentagon
    grid_z3 = np.zeros((img_size, img_size))

    for y in range(img_size):
        for x in range(img_size):
            if pentagon_matrix[y, x] == 1:
                grid_z3[y, x] = grid_z0[y, x]
            else:
                grid_z3[y, x] = str('nan')

    # choose method that you want to use
    vonMises_output = grid_z3

    # substitute Nan with zero
    vonMises_output = np.nan_to_num(vonMises_output)

    # ========================= SAVE AS .MAT FILE =================================

    # save .mat file
    #sio.savemat('stress_mat.mat', {'stress2':stress2})

    # ========================== SAVE AS IMAGE ====================================

    # save .png
    if save_png == True:

        # create empty folder
        if not os.path.exists('./images'):
            os.mkdir('./images')

        if not os.path.exists('./images/stress'):
            os.mkdir('./images/stress')

        image_temp = np.multiply(vonMises_output, 255)
        scipy.misc.imsave('./images/stress/stress{}.png'.format(image_num), image_temp)

        image_inv = Image.open('./images/stress/stress{}.png'.format(image_num))
        inverted_image = PIL.ImageOps.invert(image_inv)
        inverted_image.save('./images/stress/inv_stress{}.png'.format(image_num))

        return vonMises_output

    # ===================== VISUALIZATION OF THE METHODS ======================

#    plt.subplot(221)
#    plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
#    plt.title('Nearest')
#    plt.subplot(222)
#    plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
#    plt.title('Linear')
#    plt.subplot(223)
#    plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
#    plt.title('Cubic')
#    plt.subplot(224)
#    plt.imshow(grid_z3.T, extent=(0,1,0,1), origin='lower')
#    plt.title('NewShape')
#    plt.gcf().set_size_inches(6, 6)
#    plt.show()


###################################################################################
###################################################################################

def automate_points_stress(number_of_elements, img_size):

    # create tensor of the size: torch.Size([img_size, img_size, number_of_elements])
    in_tensor = []
    out_tensor = []

    for item in range(number_of_elements):

        # ================================= READ INPUT_PARAMS ======================
        # read the file from the folder
        pentagon_path = './input_params/InputParams{}.txt'.format(item)
        stress_path = './vonMises/VonMisesCentroid{}.txt'.format(item)

        # get the pentagon matrix
        pentagon = points_to_image(pentagon_path, 5, img_size, True, item)

        # get the stress matrix
        stress = stress_to_image(stress_path, pentagon, img_size, True, item)

        # add pentagon matrix to the stack
        in_tensor.append(pentagon)
        out_tensor.append(stress)

    # create the tensor of the size: (number_of_elements, img_size, img_size)
    # in_tensor = torch.tensor(in_tensor)
    # out_tensor = torch.tensor(out_tensor)

    sio.savemat('in_tensor.mat', {'in_tensor': in_tensor})
    sio.savemat('out_tensor.mat', {'out_tensor': out_tensor})

    return in_tensor, out_tensor


###################################################################################
###################################################################################

def get_params(pentagon_name):

    # open and read file
    file = open(pentagon_name, 'r')
    lines = list(file)
    file.close()

    img_size = 64
    polygon = 5
    number_of_parameters = 6  # X, Y, F_X, F_Y, FX, FY
    input_params = np.zeros((polygon, number_of_parameters))

    # get the data of the polygon from the InputParams.txt file
    for i in range(len(lines)):
        split_line = lines[i].split('\t')
        if (split_line[0] == 'X'):
            k = i + 1
            pos = 0
            for m in range(k, k+polygon):
                split_line = lines[m].split('\t')
                input_params[pos, 0] = float(split_line[0])
#                input_params[pos, 0] = float((np.multiply(float(split_line[0]), img_size/2)) + (img_size/2))
                input_params[pos, 1] = float(split_line[1])
#                input_params[pos, 1] = float((np.multiply(float(split_line[1]), img_size/2)) + (img_size/2))
                input_params[pos, 2] = float(split_line[3])
                input_params[pos, 3] = float(split_line[4])
                input_params[pos, 4] = float(split_line[9])
                input_params[pos, 5] = float(split_line[10])
                pos += 1

    return input_params






###################################################################################
###################################################################################

def automate_get_params(number_of_elements):

    # create empty matrix
    in_matrix = []

    for item in range(number_of_elements):

        # read the file from the folder
#        pentagon_name = './input_params/InputParams{}.txt'.format(item)
        pentagon_name = './DATA/13_MediumShape/input_params/InputParams{}.txt'.format(item)
#        pentagon_name = './DATA/08_noPressNewShape2/input_params/InputParams{}.txt'.format(item)
#        pentagon_name = './DATA/03_data_new/input_params/InputParams{}.txt'.format(item)

        # get the pentagon parameters
        pentagon_params = get_params(pentagon_name)

        # add pentagon matrix to the stack
        in_matrix.append(pentagon_params)

    # save as .mat file
    # sio.savemat('in_matirx.mat', {'in_matrix':in_matrix})

    # save as npy
    name = 'InputParams_matrix'
    np.save(name, in_matrix)

    return in_matrix


###################################################################################
###################################################################################

def get_fix_and_force(pentagon_name):

    # open and read file
    file = open(pentagon_name, 'r')
    lines = list(file)
    file.close()

    polygon = 5
    number_of_parameters = 4  # X, Y, F_X, F_Y, FX, FY
    input_params = np.zeros((polygon, number_of_parameters))

    # get the data of the polygon from the InputParams.txt file
    for i in range(len(lines)):
        split_line = lines[i].split('\t')
        if (split_line[0] == 'X'):
            k = i + 1
            pos = 0
            for m in range(k, k+polygon):
                split_line = lines[m].split('\t')
                input_params[pos, 0] = float(split_line[3])
                input_params[pos, 1] = float(split_line[4])
                input_params[pos, 2] = float(split_line[9])
                input_params[pos, 3] = float(split_line[10])
                pos += 1

    return input_params






###################################################################################
###################################################################################

def automate_get_fix_and_force(number_of_elements):

    # create empty matrix
    in_matrix = []

    for item in range(number_of_elements):

        # read the file from the folder
#        pentagon_name = './input_params/InputParams{}.txt'.format(item)
        pentagon_name = './DATA/13_MediumShape/input_params/InputParams{}.txt'.format(item)
#        pentagon_name = './DATA/08_noPressNewShape2/input_params/InputParams{}.txt'.format(item)
#        pentagon_name = './DATA/03_data_new/input_params/InputParams{}.txt'.format(item)


        # get the pentagon parameters
        pentagon_params = get_fix_and_force(pentagon_name)

        # add pentagon matrix to the stack
        in_matrix.append(pentagon_params)

    # save as .mat file
    # sio.savemat('in_matirx.mat', {'in_matrix':in_matrix})

    # save as npy
    name = 'Input_fix_and_force_matrix'
    np.save(name, in_matrix)

    return in_matrix



###################################################################################
###################################################################################

def get_pressure(pentagon_name):

    # open and read file
    file = open(pentagon_name, 'r')
    lines = list(file)
    file.close()

    pressure = []

    # get the data of the polygon from the InputParams.txt file
    for i in range(len(lines)):
        split_line = lines[i].split('\t')

        if (split_line[0] == 'pressure'):
            one_pressure = np.zeros((1, 3))
            split_line = lines[i].split('\t')
            one_pressure[0, 0] = float(split_line[1])
            one_pressure[0, 1] = float(split_line[2])
            one_pressure[0, 2] = float(split_line[3])
            pressure = np.append(pressure, one_pressure)

    pressure = np.reshape(pressure, (1, len(pressure)))

    return pressure


###################################################################################
###################################################################################

def automate_get_pressure(number_of_elements):

    # create empty matrix
    press_matrix = []

    for item in range(number_of_elements):

        # read the file from the folder
        pentagon_name = './input_params/InputParams{}.txt'.format(item)

        # get the pentagon parameters
        pentagon_press = get_pressure(pentagon_name)

        # add pentagon matrix to the stack
        press_matrix.append(pentagon_press)

    # save as .mat file
    # sio.savemat('in_matirx.mat', {'in_matrix':in_matrix})

    # save as npy
    name = 'InputPressure_matrix'
    np.save(name, press_matrix)

    return press_matrix


###################################################################################
###################################################################################

def save_as_img(matrix, img_name, colour):

    """
    matrix - matirx with the values of the image
    img_name - name of the file that we want to store
    colour - 1 or 255
    """

    # create empty folder
    if not os.path.exists('./images'):
        os.mkdir('./images')

    image_temp = np.multiply(matrix, colour)
    scipy.misc.imsave(('./images/' + img_name + '.png'), image_temp)

    image_inv = Image.open('./images/' + img_name + '.png')
    inverted_image = PIL.ImageOps.invert(image_inv)
    inverted_image.save('./images/inv' + img_name + '.png')



###################################################################################
###################################################################################

def force_to_img(params, img_size):
    pass
    
    
    
    
    
    