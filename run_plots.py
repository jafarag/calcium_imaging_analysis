# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:54:48 2022

@author: User
"""
# Plot trial averaged plot of each conditioned neuron before and during session
import os
os.getcwd()
import BCI_analysis as bci
#mat_name = ['BCI13_030222v8.mat','BCI22_030222v8.mat','.mat', '.mat', '.mat', , , , , , , , ]
for i in range(13):
    if i == 3 or i == 7:
        skip = 0;
    elif i < 9:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(i+1) + '/BCI_data/'
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-0' + str(i+1) + '/BCI_data/'
    dirpath = os.listdir(basepath)
    fullpath = basepath + dirpath[0]        
    data_dict = bci.io_matlab.read_multisession_mat(fullpath);
    bci.plot_imaging.plot_trial_averaged_trace_2sessions(data_dict,1)
    bci.plot_imaging.plot_trial_averaged_trace_2sessions(data_dict,2)
    