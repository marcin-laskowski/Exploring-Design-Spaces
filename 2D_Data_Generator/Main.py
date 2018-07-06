
# coding: utf-8

# In[4]:


#Main Generator Data

import Generate_Files_Func as GFF
import BlackBox_FE as BBFE
import GenerateDataFunctions as GDF


num_of_elements = 10
img_size = 128

# ========================= CREATE INPUT_PARAMS ===============================
GFF.Generate_n_Files(num_of_elements)

# ========================= CALCULATE VONMISES ================================
for i in range(num_of_elements):
    file_name = 'InputParams{}.txt'.format(i)
    BBFE.BlackBox_FE(file_name,i)
    print(float(i+1)/float(num_of_elements)*100,"%  of running time for BlackBox")


# ================ Generate INPUT and OUTPUT matrices =========================

[in_matrix, out_matrix] = GDF.automate_points_stress(num_of_elements, img_size)
