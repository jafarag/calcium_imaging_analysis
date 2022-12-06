# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:01:16 2022

@author: User
"""

"""
Code used for establishing signal correlation, noise correlation, residual correlations, visualization of correlation matrices, PCA, KNN imputation.
"""



import os
for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        for j in range(data_dict['n_days']):
            plt.figure()
            n_neurons = len(data_dict['dff_sessionwise_closed_loop'][j][0,:])
            first50_all = np.zeros((n_neurons,240))
            last50_all = np.zeros((n_neurons,240))
            for i in range(n_neurons):
                n_trials = len(data_dict['f_trialwise_closed_loop'][j][0,i,:])
                for m in range(n_trials):
                    if np.isnan(data_dict['f_trialwise_closed_loop'][j][:,i,m]).all() == True:
                        n_trials = m-1
                        break
                if n_trials <= 100:
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:20]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-21:n_trials-1]
                    #l50 = np.nanmean(last50[0:5,:])
                    f = np.nanmean(first50[0:5,:])
                else:
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:50]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-51:n_trials-1]
                    f = np.nanmean(first50[0:20,:])
                dff50 =  (first50- f)/f
                dfl50 =  (last50- f)/f
                first50_all[i,:] = np.nanmean(dff50,axis=1)
                last50_all[i,:] = np.nanmean(dfl50,axis=1)
                if i == data_dict['conditioned_neuron_idx'][j]:
                    plt.plot(first50_all[i,:],'b')
                else:
                    plt.plot(first50_all[i,:],'r')
                plt.plot(first50_all[data_dict['conditioned_neuron_idx'][j],:],'b')
                plt.xlabel('Time points')
                plt.ylabel('ΔF/F')
            plt.title('first 50 trial-averaged values for n=' + str(n_neurons) + ' neurons, blue = neuron ' + str(data_dict['conditioned_neuron_idx'][j]))
            plt.savefig('first 50 trials session' + str(j) + ' dF-f')
    
    
#next step: try to look at teh difference between first 50 and last fifty trials between signals (df/f)
#for each of the neurons

import os
for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        for j in range(data_dict['n_days']):
            plt.figure()
            n_neurons = len(data_dict['dff_sessionwise_closed_loop'][j][0,:])
            diff50_all = np.zeros((n_neurons,201))
            for i in range(n_neurons):
                n_trials = len(data_dict['f_trialwise_closed_loop'][j][0,i,:])
                for m in range(n_trials):
                    if np.isnan(data_dict['f_trialwise_closed_loop'][j][39:,i,m]).all() == True:
                        n_trials = m-1
                        break
                if ((k == 0 and j == 1) or (n_trials <= 100)):
                    first50 = data_dict['f_trialwise_closed_loop'][j][39:,i,0:20]
                    last50 = data_dict['f_trialwise_closed_loop'][j][39:,i,n_trials-21:n_trials-1]
                else:
                    first50 = data_dict['f_trialwise_closed_loop'][j][39:,i,0:50]
                    last50 = data_dict['f_trialwise_closed_loop'][j][39:,i,n_trials-51:n_trials-1]
                #f = np.nanmean(last50[0:20,:])
                dff =  (last50-first50)/first50
                diff50_all[i,:] = np.nanmean(dff,axis=1)
                if i == data_dict['conditioned_neuron_idx'][j]:
                    plt.plot(diff50_all[i,:],'b')
                else:
                    plt.plot(diff50_all[i,:],'r')
                plt.plot(diff50_all[data_dict['conditioned_neuron_idx'][j],:],'b')
                plt.xlabel('Time points')
                plt.ylabel('ΔF/F')
            plt.title('Trial start last 50 - first 50 trial-averaged values for n=' + str(n_neurons) + ' neurons, blue = neuron ' + str(data_dict['conditioned_neuron_idx'][j]))
            plt.savefig('Trial start last 50 - first 50 trials session' + str(j) + ' dF-f')


# plot only for modualted neuron


import os
for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        mod_neuron = np.zeros((data_dict['n_days'],240))
        for j in range(data_dict['n_days']):
            plt.figure()
            n_neurons = len(data_dict['dff_sessionwise_closed_loop'][j][0,:])
            diff50_all = np.zeros((n_neurons,240))
            for i in range(n_neurons):
                n_trials = len(data_dict['f_trialwise_closed_loop'][j][0,i,:])
                for m in range(n_trials):
                    if np.isnan(data_dict['f_trialwise_closed_loop'][j][:,i,m]).all() == True:
                        n_trials = m-1
                        break
                if ((k == 0 and j == 1) or (n_trials <= 100)):
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:20]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-21:n_trials-1]
                else:
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:50]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-51:n_trials-1]
                #f = np.nanmean(last50[0:20,:])
                dff =  (last50-first50)/first50
                diff50_all[i,:] = np.nanmean(dff,axis=1)
                if i == data_dict['conditioned_neuron_idx'][j]:
                    mod_neuron[j,:] = diff50_all[i,:]
                    plt.plot(diff50_all[i,:],'b')
                plt.xlabel('Time points')
                plt.ylabel('ΔF/F')
            plt.title('last 50 - first 50 trial-averaged values for modulated neuron = ' + str(data_dict['conditioned_neuron_idx'][j]))
            plt.savefig('last 50 - first 50 trials session' + str(j) + ' dF-f for neuron ' + str(data_dict['conditioned_neuron_idx'][j]))


# new way for plotting df/f

import os
mods = np.zeros((8,1),dtype=object)
for k in [0,1,4,6]:
    basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
    os.chdir(basepath)
    dirpath = os.listdir(basepath)
    fullpath = basepath + dirpath[0]        
    data_dict = bci.io_matlab.read_multisession_mat(fullpath);
    mod_neuron = np.zeros((data_dict['n_days'],240))
    for j in range(data_dict['n_days']):
        plt.figure()
        n_neurons = len(data_dict['dff_sessionwise_closed_loop'][j][0,:])
        diff50_all = np.zeros((n_neurons,240))
        for i in range(n_neurons):
            n_trials = len(data_dict['f_trialwise_closed_loop'][j][0,i,:])
            for m in range(n_trials):
                if np.isnan(data_dict['f_trialwise_closed_loop'][j][:,i,m]).all() == True:
                    n_trials = m-1
                    break
            if ((k == 0 and j == 1) or (n_trials <= 100)):
                first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:20]
                last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-21:n_trials-1]
            else:
                first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:50]
                last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-51:n_trials-1]
            #f = np.nanmean(last50[0:20,:])
            dff =  (last50/first50) - (first50/first50)
            diff50_all[i,:] = np.nanmean(dff,axis=1)
            if i == data_dict['conditioned_neuron_idx'][j]:
                mod_neuron[j,:] = diff50_all[i,:]
                plt.plot(diff50_all[i,:],'b')
            else:
                plt.plot(diff50_all[i,:],'r') 
            plt.plot(diff50_all[data_dict['conditioned_neuron_idx'][j],:],'b')
            plt.xlabel('Time points')
            plt.ylabel('ΔF/F')
        plt.title('last 50 - first 50 trial-averaged values for modulated neuron = ' + str(data_dict['conditioned_neuron_idx'][j]) + 'NEW method')
        plt.savefig('last 50 - first 50 trials session' + str(j) + ' dF-f for neuron ' + str(data_dict['conditioned_neuron_idx'][j]) + 'NEW method')




    
first50_all = np.zeros((len(data_dict['dff_sessionwise_closed_loop'][1][0,:]),240))
last50_all = np.zeros((len(data_dict['dff_sessionwise_closed_loop'][1][0,:]),240))
for i in range(len(data_dict['dff_sessionwise_closed_loop'][1][0,:])):
    n_trials = len(data_dict['f_trialwise_closed_loop'][1][0,i,:])
    first50 = data_dict['f_trialwise_closed_loop'][1][:,i,0:49]
    last50 = data_dict['f_trialwise_closed_loop'][1][:,i,n_trials-51:n_trials-1]
    first50_all[i,:] = np.mean(first50,axis=1)
    last50_all[i,:] = np.mean(last50,axis=1)  
    
#mean then take difference    
    
import os
for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        for j in range(data_dict['n_days']):
            plt.figure()
            n_neurons = len(data_dict['dff_sessionwise_closed_loop'][j][0,:])
            diff50_all = np.zeros((n_neurons,240))
            for i in range(n_neurons):
                n_trials = len(data_dict['f_trialwise_closed_loop'][j][0,i,:])
                for m in range(n_trials):
                    if np.isnan(data_dict['f_trialwise_closed_loop'][j][:,i,m]).all() == True:
                        n_trials = m-1
                        break
                if ((k == 0 and j == 1) or (n_trials <= 100)):
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:20]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-21:n_trials-1]
                else:
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:50]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-51:n_trials-1]
                #f = np.nanmean(last50[0:20,:])
                dff = np.nanmean(last50/first50,axis=1)-np.nanmean(first50/first50,axis=1)
                diff50_all[i,:] = dff
                plt.plot(diff50_all[i,:],'r')
                plt.plot(diff50_all[data_dict['conditioned_neuron_idx'][j],:],'b')
                plt.xlabel('Time points')
                plt.ylabel('ΔF/F')
            plt.title('last 50 - first 50 trial-averaged values for n=' + str(n_neurons) + ' neurons, blue = neuron ' + str(data_dict['conditioned_neuron_idx'][j]))
            plt.savefig('last 50 - first 50 trials session' + str(j) + ' dF-f mean first NEW only during norm')
            
            
    # normalize first by the first 50 trials (first few time points of trials) and do the same by last 50 trials 
    
import os
for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        for j in range(data_dict['n_days']):
            plt.figure()
            n_neurons = len(data_dict['dff_sessionwise_closed_loop'][j][0,:])
            diff50_all = np.zeros((n_neurons,240))
            for i in range(n_neurons):
                n_trials = len(data_dict['f_trialwise_closed_loop'][j][0,i,:])
                for m in range(n_trials):
                    if np.isnan(data_dict['f_trialwise_closed_loop'][j][:,i,m]).all() == True:
                        n_trials = m-1
                        break
                if ((k == 0 and j == 1) or (n_trials <= 100)):
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:20]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-21:n_trials-1]
                    f = np.nanmean(first50[0:5,:])
                    f50 = first50/f
                    l50 = last50/f
                else:
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:50]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-51:n_trials-1]
                    f = np.nanmean(first50[0:20,:])
                    f50 = first50/f
                    l50 = last50/f
                dff = np.nanmean(l50-f50,axis=1)
                diff50_all[i,:] = dff
                plt.plot(diff50_all[i,:],'r')
                plt.plot(diff50_all[data_dict['conditioned_neuron_idx'][j],:],'b')
                plt.xlabel('Time points')
                plt.ylabel('ΔF/F')
            plt.title('last 50 - first 50 trial-averaged values for n=' + str(n_neurons) + ' neurons, blue = neuron ' + str(data_dict['conditioned_neuron_idx'][j]))
            plt.savefig('last 50 - first 50 trials session' + str(j) + ' dF-f norm by first, only subtraction')
            
            


# first50 and last50 noise correlations for each neuron, plot into matrix

import os
for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        for j in range(data_dict['n_days']):
            plt.figure()
            n_neurons = len(data_dict['dff_sessionwise_closed_loop'][j][0,:])
            first50_all = np.zeros((n_neurons,240))
            last50_all = np.zeros((n_neurons,240))
            for i in range(n_neurons):
                n_trials = len(data_dict['f_trialwise_closed_loop'][j][0,i,:])
                for m in range(n_trials):
                    if np.isnan(data_dict['f_trialwise_closed_loop'][j][:,i,m]).all() == True:
                        n_trials = m-1
                        break
                if n_trials <= 100:
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:20]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-21:n_trials-1]
                    l50 = np.nanmean(last50[0:5,:])
                    f50 = np.nanmean(first50[0:5,:])
                else:
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:50]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-51:n_trials-1]
                    f50 = np.nanmean(first50[0:20,:])
                    l50 = np.nanmean(last50[0:20,:])
                dff50 =  (first50- f50)/f50
                dfl50 =  (last50- l50)/l50
                first50_all[i,:] = np.nanmean(dff50,axis=1)
                last50_all[i,:] = np.nanmean(dfl50,axis=1)
            #corr_mat = np.corrcoef(first50_all)
            #sns.heatmap(corr_mat, vmin=-1, vmax=1, center=0)
            last50_corr = np.corrcoef(last50_all)
            sns.heatmap(last50_corr, vmin=-1, vmax=1, center=0)
            #plt.plot(first50_all[data_dict['conditioned_neuron_idx'][j],:],'b')
            plt.xlabel('Neurons')
            plt.ylabel('Neurons')
            plt.title('Signal corr first 50 trials for n=' + str(n_neurons) + ' neurons, blue = neuron ' + str(data_dict['conditioned_neuron_idx'][j]))
            plt.savefig('Signal correlations last 50 trials session' + str(j) + ' dF-f')
            
            
#noise correlations
import os
for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        for j in range(data_dict['n_days']):
            plt.figure()
            n_neurons = len(data_dict['dff_sessionwise_closed_loop'][j][0,:])
            n_trials = len(data_dict['f_trialwise_closed_loop'][j][0,i,:])
            for m in range(n_trials):
                if np.isnan(data_dict['f_trialwise_closed_loop'][j][:,i,m]).all() == True:
                    n_trials = m-1
                    break
            if n_trials <= 100:
                n_use = 20
            else:
                n_use = 50
            first50_all = np.zeros((n_neurons,240,n_use),dtype=object)
            last50_all = np.zeros((n_neurons,240,n_use),dtype=object)
            for i in range(n_neurons):
                n_trials = len(data_dict['f_trialwise_closed_loop'][j][0,i,:])
                for m in range(n_trials):
                    if np.isnan(data_dict['f_trialwise_closed_loop'][j][:,i,m]).all() == True:
                        n_trials = m-1
                        break
                if n_trials <= 100:
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:20]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-21:n_trials-1]
                    l50 = np.nanmean(last50[0:5,:])
                    f50 = np.nanmean(first50[0:5,:])
                else:
                    first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:50]
                    last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-51:n_trials-1]
                    f50 = np.nanmean(first50[0:20,:])
                    l50 = np.nanmean(last50[0:20,:])
                dff50 =  (first50- f50)/f50
                dfl50 =  (last50- l50)/l50
                first50_all[i,:,:] = dff50
                last50_all[i,:,:] = dfl50
            #corr_mat = np.corrcoef(first50_all)
            #sns.heatmap(corr_mat, vmin=-1, vmax=1, center=0)
            last50_corrs = signal.correlate(last50_all,last50_all)
            sns.heatmap(last50_corrs, vmin=-1, vmax=1, center=0)
            #plt.plot(first50_all[data_dict['conditioned_neuron_idx'][j],:],'b')
            plt.xlabel('Neurons')
            plt.ylabel('Neurons')
            plt.title('Noise corr last 50 trials for n=' + str(n_neurons) + ' neurons, blue = neuron ' + str(data_dict['conditioned_neuron_idx'][j]))
            plt.savefig('Noise correlations last 50 trials session' + str(j) + ' dF-f')




#noise correlation calculation:
def noise_corr(x,y):
    #Calculate noise correlation between normalized x and y
    #Generate noise correlation matrix between all combinations
    #Assumption: x and y must be same length and normalized
    n_corr_mat = np.zeros((x.shape[0],y.shape[0]))
    nx = x.shape[1]
    lags = np.arange(-nx + 1, nx)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
             n_corr_mat[i,j] = lags[np.argmax(np.correlate(x[i,:], y[j], mode='full'))]
    return n_corr_mat


#signalcorrelation calculation:
def sig_corr(x,y):
    #Calculate signal correlation between normalized x and y
    #Generate signal correlation matrix between all combinations
    #Assumption: x and y must be same length and normalized
    n_corr_mat = np.zeros((x.shape[0],y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
             n_corr_mat[i,j] = np.min(np.corrcoef(x[i,:], y[j,:]))
    return n_corr_mat


#do before-before, after-after, and before-after matrix
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(x, y)
axs[1].plot(x, -y)
 
neur = np.concat(np.linspace(0,n_neurons),np.linspace(0,n_neurons))

#plot diagonal values of first v last 50
y_first = np.arange(0,n_neurons-1)
x_last = np.arange(n_neurons,2*(n_neurons-1))
plt.plot(firstla[x_last,y_first])




import os

for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        for j in range(data_dict['n_days']):
#function to compute the average of the first 50 or last 50 
def average_trials_per_session(data_dict,j):
    plt.figure()
    n_neurons = len(data_dict['dff_sessionwise_closed_loop'][j][0,:])
    for i in range(n_neurons):
        n_trials = len(data_dict['f_trialwise_closed_loop'][j][0,i,:])
        for m in range(n_trials):
            if np.isnan(data_dict['f_trialwise_closed_loop'][j][:,i,m]).all() == True:
                n_trials = m-1
                break
       # if i==n_neurons-1:
        #    print(n_trials)
        #if n_trials <= 100:           
        #good_trials = data_dict['f_trialwise_closed_loop'][j][39:,i,:n_trials]
        first50_all = np.zeros((n_neurons,201))
        last50_all = np.zeros((n_neurons,201))
        #print(good_trials.shape)
    for k in range(n_neurons):
        last50 = data_dict['f_trialwise_closed_loop'][j][39:,k,n_trials-21:n_trials-1]
        first50 = data_dict['f_trialwise_closed_loop'][j][39:,k,0:20]
        #print('-NEW-')
        #print('last50')
        l50 = np.nanmean(last50[0:5,:])
        #print(l50)
        #print('first50')
        f50 = np.nanmean(first50[0:5,:])
        #print(f50)
        """
            else:
                first50 = data_dict['f_trialwise_closed_loop'][j][:,i,0:50]
                last50 = data_dict['f_trialwise_closed_loop'][j][:,i,n_trials-51:n_trials-1]
                f50 = np.nanmean(first50[0:20,:])
                l50 = np.nanmean(last50[0:20,:])
                print('-NEW-')
                print('last50')
                l50 = np.nanmean(last50[0:5,:])
                print(l50)
                print('first50')
                f50 = np.nanmean(first50[0:5,:])
                print(f50)
        """
        dff50 =  (first50- f50)/f50
        dfl50 =  (last50- l50)/l50
        first50_all[k,:] = np.nanmean(dff50,axis=1)
        last50_all[k,:] = np.nanmean(dfl50,axis=1)
    return first50_all, last50_all

#function to plot each correlation matrix for first+last cross correlation as well as first v last trace plot
for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        for j in range(data_dict['n_days']):
            first50_all, last50_all = average_trials_per_session(data_dict,j)
            firlas_all = np.concatenate((first50_all,last50_all))
            firlas_corr_n = np.corrcoef(firlas_all)
            sns.heatmap(firlas_corr_n, vmin=-1, vmax=1, center=0)
            n_neurons = len(data_dict['dff_sessionwise_closed_loop'][j][0,:])
            
            plt.title('First vs last trials, n=' + str(n_neurons) + ' neurons, modulated neuron =' + str(data_dict['conditioned_neuron_idx'][j]))
            plt.xlabel('Neurons (0-' + str(n_neurons-1) + ' = first,' + str(n_neurons) + '-' + str((2*n_neurons)-1) + ' = last)')
            plt.ylabel('Neurons (0-' + str(n_neurons-1) + ' = first,' + str(n_neurons) + '-' + str((2*n_neurons)-1) + ' = last)')
            plt.savefig('Trial start n=20 trials DFF f+l first vs last trial-averaged correlation session ' + str(j) + ' df-f only first')
            
            firstla = sig_corr(first50_all,last50_all)
            y_first = np.arange(0,n_neurons-1)
            
            plt.figure()
            colors = ["b" if q == data_dict['conditioned_neuron_idx'][j] else "r" for q in range(n_neurons-1)]
            plt.bar(y_first,firstla[y_first,y_first],color=colors)        
            plt.title('First -last r-values for n=' + str(n_neurons) + ' neurons, modulated neuron =' + str(data_dict['conditioned_neuron_idx'][j]))
            plt.xlabel('Neurons (0-' + str(n_neurons-1) + ')')
            plt.ylabel('r-value (Correlation of first trials & end trials responses)')
            plt.savefig('Trial start n=20 trials DFF f+l first & last trial r-values for each neuron session ' + str(j)) 
            
            plt.figure()
            no_mod_y = np.delete(y_first,data_dict['conditioned_neuron_idx'][j])
            #C_ij = firstla[27,no_mod_y] #before after, modulated neuron with everyone else
            #C_oi = firlas_corr_n[27,no_mod_y] #before before, modulated neuron 
            C_ij = firstla[data_dict['conditioned_neuron_idx'][j],no_mod_y].reshape(firstla.shape[0]-2, 1)
            C_oi = firlas_corr_n[data_dict['conditioned_neuron_idx'][j],no_mod_y].reshape(firstla.shape[0]-2, 1)
            C_oi_d = np.append(C_oi, np.ones((firstla.shape[0]-2, 1)), axis=1)
            theta = np.linalg.inv(C_oi_d.T.dot(C_oi_d)).dot(C_oi_d.T).dot(C_ij)
            y_line = C_oi_d.dot(theta)
            #colors = ["b" if q == data_dict['conditioned_neuron_idx'][j] else "r" for q in range(n_neurons-1)]
            plt.scatter(C_oi,C_ij)
            plt.plot(C_oi,y_line,'r')
            plt.title('$C_{oi}$ vs $C_{ij}$, r = ' + str(round(theta[0][0],3)) + ' for n=' + str(n_neurons) + ' neurons, mod neuron =' + str(data_dict['conditioned_neuron_idx'][j]))
            plt.xlabel('$C_{oi}$ r-value')
            plt.ylabel('$C_{ij}$ r-value')
            plt.savefig('Trial start Before-before vs Before-after r-values for modulated neuron session ' + str(j))   
                
                
dt = np.array([
          [0.05, 0.11],
          [0.13, 0.14],
          [0.19, 0.17],
          [0.24, 0.21],
          [0.27, 0.24],
          [0.29, 0.32],
          [0.32, 0.30],
          [0.36, 0.39],
          [0.37, 0.42],
          [0.40, 0.40],
          [0.07, 0.09],
          [0.02, 0.04],
          [0.15, 0.19],
          [0.39, 0.32],
          [0.43, 0.48],
          [0.44, 0.41],
          [0.47, 0.49],
          [0.50, 0.57],
          [0.53, 0.59],
          [0.57, 0.51],
          [0.58, 0.60]
])
x = dt[:, 0].reshape(dt.shape[0], 1)
X = np.append(x, np.ones((dt.shape[0], 1)), axis=1)
y = dt[:, 1].reshape(dt.shape[0], 1)

# Calculating the parameters using the least square method
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(f'The parameters of the line: {theta}')

# Now, calculating the y-axis values against x-values according to
# the parameters theta0 and theta1
y_line = C_oi_d.dot(theta)

# Plotting the data points and the best fit line
plt.scatter(x, y)
plt.plot(x, y_line, 'r')


    

C_ij = firstla[27,y_first]
C_oi = firlas_corr_n[27,y_first]
plt.scatter(x=C_ij,y=C_oi)


#actual noise correlations:
#Step1: Take each neurons normalized response and bin by each time point (1:240), 
#though you can change bin size later
#Step2: compute noise correlation for each trial at that time point per pairs of neurons at that specific timepoint:
#Step3: compute 
import numpy as np
import numpy.ma as ma
def noise_correlation(dat_r,b_size=1,ses=-1):
    """ INPUTS 
    dat_r = raw data_dict with all trials, usually use f_sessionwise
    b_size = bin size of the time series, defaulted to 1
    ses = session number in trial
    """    
    n_neuron = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    real_neur = norm_trial_f_neuron(dat_r,ses)
    neur2_time_r = np.zeros((n_neuron,n_neuron,240))
    n_pair = n_neuron
    unique_pair = 0
    np.arange(0,240,b_size)
    for t in range (240):
        for i in range(n_neuron):
            n1_trials = real_neur[i,t,:]
            n1=ma.masked_invalid(n1_trials)
            for j in range(n_neuron):
                n2_trials = real_neur[j,t,:]
                n2=ma.masked_invalid(n2_trials)
                msk = (~n1.mask & ~n2.mask)
                neur2_time_r[i,j,t] = np.correlate(n1_trials[msk],n2_trials[msk])
    return neur2_time_r
                

import numpy as np
import numpy.ma as ma
import scipy.stats as stat
def noise_correlation_RESHAPE(dat_r,b_size=1,ses=-1):
    """ INPUTS 
    dat_r = raw data_dict with all trials, usually use f_sessionwise
    b_size = bin size of the time series, defaulted to 1
    ses = session number in trial
    """    
    n_neuron = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    real_neur = norm_trial_f_neuron(dat_r,ses)
    neur2_time_r = np.zeros((n_neuron,n_neuron))
    #n_pair = n_neurona
    #unique_pair = 0
    #np.arange(0,240,b_size)
    for i in range(n_neuron):
        n1_trials = real_neur[i,:,:].flatten()
        n1=ma.masked_invalid(n1_trials)
        for j in range(n_neuron):
            n2_trials = real_neur[j,:,:].flatten()
            n2=ma.masked_invalid(n2_trials)
            msk = (~n1.mask & ~n2.mask)
            neur2_time_r[i,j] = stat.pearsonr(n1_trials[msk],n2_trials[msk])[0]
    return neur2_time_r            



                
"""                
            r_noise
        while n_pair >=1:
            unique_pair += n_pair
            n_pair -=
    r_noise = np.zeros((unique_pair,240))
"""   

overshoot = np.array([0,0,1,3,5,50,55,65,75,75,75,0])
ap = np.array([0,0,0,0,0,1,1,2,3,3,1,0])

def norm_trial_f_neuron(dat_r,ses):
    plt.figure()
    n_neurons = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    #n_tri = np.zeros((n_neurons,1))
    for i in range(n_neurons):
        if i == 0:
            n_trials = len(data_dict['f_trialwise_closed_loop'][ses][0,i,:])
        else:
            for m in range(n_trials):
                if np.isnan(data_dict['f_trialwise_closed_loop'][ses][39:,i,m]).all() == True:
                    n_trials = m-1
                    break
    first50_all = np.zeros((n_neurons,201,n_trials))
    for q in range(n_neurons):
        good_trials = data_dict['f_trialwise_closed_loop'][ses][39:,q,:n_trials]
        print(good_trials.shape)
        for k in range(n_trials):
            f50 = np.nanmean(good_trials[0:5,k])
            dff50 =  (good_trials[:,k]- f50)/f50
            #dfl50 =  (last50- l50)/l50
            first50_all[q,:,k] = dff50
            #last50_all[i,:] = np.nanmean(dfl50,axis=1)
    return first50_all #, last50_all
        
def noise_corr_REAL(xy,n1,n2):
    # xy is real_neur (shape = neurons,trial timepoints, trials)  
    x = xy[n1,:,:].reshape(xy.shape[1]*xy.shape[2], 1)
    print(x.shape)
    y = xy[n2,:,:].reshape(xy.shape[1]*xy.shape[2], 1)
    print(y.shape)
    x_ma=ma.masked_invalid(x)
    y_ma=ma.masked_invalid(y)
    msk = (~x_ma.mask & ~y_ma.mask)
    X = np.append(x_ma, np.ones((x_ma.shape[0], 1)), axis=1)
    print(X.shape)
    #print(X.shape)
    #n1=ma.masked_invalid(n1_trials)
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_ma)
    return theta

# Calculating the parameters using the least square method
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


y_first = np.arange(0,n_neurons-1)
no_mod_y = np.delete(y_first,data_dict['conditioned_neuron_idx'][-1])
#C_ij = firstla[27,no_mod_y] #before after, modulated neuron with everyone else
#C_oi = firlas_corr_n[27,no_mod_y] #before before, modulated neuron 
C_ij = noise_mat_RESHAPE[data_dict['conditioned_neuron_idx'][-1],no_mod_y].reshape(firstla.shape[0]-2, 1)
C_oi = noise_mat_RESHAPE[data_dict['conditioned_neuron_idx'][-1],no_mod_y].reshape(firstla.shape[0]-2, 1)
C_oi_d = np.append(C_oi, np.ones((firstla.shape[0]-2, 1)), axis=1)
theta = np.linalg.inv(C_oi_d.T.dot(C_oi_d)).dot(C_oi_d.T).dot(C_ij)
y_line = C_oi_d.dot(theta)
#colors = ["b" if q == data_dict['conditioned_neuron_idx'][j] else "r" for q in range(n_neurons-1)]
plt.scatter(C_oi,C_ij)
plt.plot(C_oi,y_line,'r')
plt.title('$C_{oi}$ vs $C_{ij}$, r = ' + str(round(theta[0][0],3)) + ' for n=' + str(n_neurons) + ' neurons, mod neuron =' + str(data_dict['conditioned_neuron_idx'][j]))
plt.xlabel('$C_{oi}$ r-value')
plt.ylabel('$C_{ij}$ r-value')
plt.savefig('Before-before vs Before-after r-values for modulated neuron session ' + str(j))


y_first = np.arange(0,n_neurons-1)
plt.figure()
colors = ["b" if q == data_dict['conditioned_neuron_idx'][-1] else "r" for q in range(n_neurons-1)]
plt.bar(y_first,noise_mat_RESHAPE[data_dict['conditioned_neuron_idx'][-1],y_first],color=colors) 
plt.ylim =(-0.2,0.4)
plt.title('Noise correlation (all trials) r-values for n=' + str(n_neurons) + ' neurons, modulated neuron =' + str(data_dict['conditioned_neuron_idx'][-1]))
plt.xlabel('Neurons (0-' + str(n_neurons-1) + ')')
plt.ylabel('r-value (Noise correlation all trials)')
plt.savefig('Noise correlation (all trials) r-values mod neuron' + str(data_dict['conditioned_neuron_idx'][-1]) + ' for each neuron session ' + str(-1))



#take the residual correlations of the data
#avg all trials per neuron, subtract each trial from average
#save this as residuals
#do same correlations as before
def residuals(dat_r,ses):
    n_neurons = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    #n_tri = np.zeros((n_neurons,1))
    for i in range(n_neurons):
        if i == 0:
            n_trials = len(data_dict['f_trialwise_closed_loop'][ses][0,i,:])
        else:
            for m in range(n_trials):
                if np.isnan(data_dict['f_trialwise_closed_loop'][ses][:,i,m]).all() == True:
                    n_trials = m-1
                    break
    first50_all = np.zeros((n_neurons,240,n_trials))
    residual_neur = np.zeros((n_neurons,240,n_trials))
    for q in range(n_neurons):
        good_trials = data_dict['f_trialwise_closed_loop'][ses][:,q,:n_trials]
        #print(good_trials.shape)
        for k in range(n_trials):
            f50 = np.nanmean(good_trials[0:5,k])
            dff50 =  (good_trials[:,k]- f50)/f50
            #dfl50 =  (last50- l50)/l50
            first50_all[q,:,k] = dff50
        first50_mean = np.nanmean(first50_all[q,:,:],axis=1)
        for n in range(n_trials):
             residual_neur[q,:,n] = first50_all[q,:,n] - first50_mean
    return residual_neur
            #last50_all[i,:] = np.nanmean(dfl50,axis=1)

def residual_correlations(dat_r,b_size=1,ses=-1):
    """
    TO DO: takes residuals (subtract mean) for each trial/neuron and gives corr matrix of it
    INPUTS 
    dat_r = raw data_dict with all trials, usually use f_sessionwise
    b_size = bin size of the time series, defaulted to 1
    ses = session number in trial
    """    
    n_neuron = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    real_neur = residuals(dat_r,ses)
    neur2_time_r = np.zeros((n_neuron,n_neuron))
    #n_pair = n_neurona
    #unique_pair = 0
    #np.arange(0,240,b_size)
    for i in range(n_neuron):
        n1_trials = real_neur[i,:,:].flatten()
        n1=ma.masked_invalid(n1_trials)
        for j in range(n_neuron):
            n2_trials = real_neur[j,:,:].flatten()
            n2=ma.masked_invalid(n2_trials)
            msk = (~n1.mask & ~n2.mask)
            neur2_time_r[i,j] = stat.pearsonr(n1_trials[msk],n2_trials[msk])[0]
    return neur2_time_r

def plot_resid_corr(res_neur,dat_r,ses=-1):
    n_neurons = n_neurons = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    y_first = np.arange(0,n_neurons-1)
    plt.figure()
    colors = ["r" for q in range(n_neurons-1)]
    no_mod_y = np.delete(y_first,data_dict['conditioned_neuron_idx'][-1])
    plt.bar(no_mod_y,res_neur[dat_r['conditioned_neuron_idx'][-1],no_mod_y],color=colors) 
    plt.ylim =(-0.2,0.4)
    plt.title('Residual correlation (all trials) r-values for n=' + str(n_neurons) + ' neurons, modulated neuron =' + str(dat_r['conditioned_neuron_idx'][ses]))
    plt.xlabel('Neurons (0-' + str(n_neurons-1) + ')')
    plt.ylabel('r-value (Residual correlation all trials)')
    plt.savefig('Residual correlation (all trials) r-values mod neuron' + str(dat_r['conditioned_neuron_idx'][ses]) + ' for each neuron session ' + str(ses))
    return plt.show()
    
    
    #Try again, but without normalizing the data first per trial, and only do it with all of the trials
    
def residuals_NORM(dat_r,ses):
    n_neurons = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    #n_tri = np.zeros((n_neurons,1))
    for i in range(n_neurons):
        if i == 0:
            n_trials = len(data_dict['f_trialwise_closed_loop'][ses][0,i,:])
        else:
            for m in range(n_trials):
                if np.isnan(data_dict['f_trialwise_closed_loop'][ses][:,i,m]).all() == True:
                    n_trials = m-1
                    break
    first50_all = np.zeros((n_neurons,240,n_trials))
    residual_neur = np.zeros((n_neurons,240,n_trials))
    for q in range(n_neurons):
        good_trials = data_dict['f_trialwise_closed_loop'][ses][:,q,:n_trials]
        #print(good_trials.shape)
        first50_all[q,:,:] = good_trials
        first50_mean = np.nanmean(first50_all[q,:,:],axis=1)
        for n in range(n_trials):
             residual_neur[q,:,n] = first50_all[q,:,n] - first50_mean
    return residual_neur
            #last50_all[i,:] = np.nanmean(dfl50,axis=1)

def residual_correlations_NORM(dat_r,b_size=1,ses=-1):
    """
    TO DO: takes residuals (subtract mean) for each trial/neuron and gives corr matrix of it
    INPUTS 
    dat_r = raw data_dict with all trials, usually use f_sessionwise
    b_size = bin size of the time series, defaulted to 1
    ses = session number in trial
    """    
    n_neuron = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    real_neur = residuals_NORM(dat_r,ses)
    neur2_time_r = np.zeros((n_neuron,n_neuron))
    #n_pair = n_neurona
    #unique_pair = 0
    #np.arange(0,240,b_size)
    for i in range(n_neuron):
        n1_trials = real_neur[i,:,:].flatten()
        n1=ma.masked_invalid(n1_trials)
        for j in range(n_neuron):
            n2_trials = real_neur[j,:,:].flatten()
            n2=ma.masked_invalid(n2_trials)
            msk = (~n1.mask & ~n2.mask)
            neur2_time_r[i,j] = stat.pearsonr(n1_trials[msk],n2_trials[msk])[0]
    return neur2_time_r

def plot_resid_corr_NORM(res_neur,dat_r,ses=-1):
    n_neurons = n_neurons = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    y_first = np.arange(0,n_neurons-1)
    plt.figure()
    colors = ["r" for q in range(n_neurons-1)]
    no_mod_y = np.delete(y_first,data_dict['conditioned_neuron_idx'][-1])
    plt.bar(no_mod_y,res_neur[dat_r['conditioned_neuron_idx'][-1],no_mod_y],color=colors) 
    plt.ylim =(-0.2,0.4)
    plt.title('Residual correlation (all trials) r-values for n=' + str(n_neurons) + ' neurons, modulated neuron =' + str(dat_r['conditioned_neuron_idx'][ses]))
    plt.xlabel('Neurons (0-' + str(n_neurons-1) + ')')
    plt.ylabel('r-value (Residual correlation all trials)')
    plt.savefig('Residual correlation (all trials) r-values mod neuron' + str(dat_r['conditioned_neuron_idx'][ses]) + ' for each neuron session ' + str(ses))
    
    
#What to do if no noise correlation, yet we have found changes between the beginning of the session and 
#the end of the session. Do we see whether noise correlations in first and last twenty sessions makes a 
#difference? If the changes across the entire session are less correlated to other types

#Option 1: Segment the parts of the trial prior to licking aka when the learning happens during the trial
#
#
#
#
#

#Try doing noise correlations for first v last 20 trials
#included variable trial number
def residuals_NORM_fl(dat_r,ses,trial_num):
    n_neurons = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    #n_tri = np.zeros((n_neurons,1))
    for i in range(n_neurons):
        if i == 0:
            n_trials = len(data_dict['f_trialwise_closed_loop'][ses][0,i,:])
        else:
            for m in range(n_trials):
                if np.isnan(data_dict['f_trialwise_closed_loop'][ses][:,i,m]).all() == True:
                    n_trials = m-1
                    break
    last20_all = np.zeros((n_neurons,240,trial_num))
    first20_all = np.zeros((n_neurons,240,trial_num))
    residual_neur_first = np.zeros((n_neurons,240,trial_num))
    residual_neur_last = np.zeros((n_neurons,240,trial_num))
    for q in range(n_neurons):
        first20_all[q,:,:] = dat_r['f_trialwise_closed_loop'][ses][:,q,0:trial_num]
        last20_all[q,:,:] = dat_r['f_trialwise_closed_loop'][ses][:,q,n_trials-(trial_num+1):n_trials-1]
        first20_mean = np.nanmean(first20_all[q,:,:],axis=1)
        last20_mean = np.nanmean(last20_all[q,:,:],axis=1)
        for n in range(trial_num):
             residual_neur_first[q,:,n] = first20_all[q,:,n] - first20_mean
             residual_neur_last[q,:,n] = last20_all[q,:,n] - last20_mean
    residual_neur_firstlast = np.concatenate((residual_neur_first,residual_neur_last))
    return residual_neur_firstlast
            #last50_all[i,:] = np.nanmean(dfl50,axis=1)

def residual_correlations_NORM_fl(dat_r,b_size=1,ses=-1):
    """
    TO DO: takes residuals (subtract mean) for each trial/neuron and gives corr matrix of it
    INPUTS 
    dat_r = raw data_dict with all trials, usually use f_sessionwise
    b_size = bin size of the time series, defaulted to 1
    ses = session number in trial
    """    
    n_neuron = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    real_neur = residuals_NORM_fl(dat_r,ses)
    neur2_time_r = np.zeros((2*n_neuron,2*n_neuron))
    #n_pair = n_neurona
    #unique_pair = 0
    #np.arange(0,240,b_size)
    for i in range(2*n_neuron):
        n1_trials = real_neur[i,:,:].flatten()
        n1=ma.masked_invalid(n1_trials)
        for j in range(2*n_neuron):
            n2_trials = real_neur[j,:,:].flatten()
            n2=ma.masked_invalid(n2_trials)
            msk = (~n1.mask & ~n2.mask)
            neur2_time_r[i,j] = stat.pearsonr(n1_trials[msk],n2_trials[msk])[0]
    return neur2_time_r

def plot_resid_corr_NORM_fl(res_neur,dat_r,ses=-1):
    n_neurons = n_neurons = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    y_first = np.arange(0,n_neurons-1)
    x_last = np.arange(n_neurons,(2*n_neurons)-1)
    plt.figure()
    colors = ["b" if q == dat_r['conditioned_neuron_idx'][-1] else "r" for q in range(n_neurons-1)]
    no_mod_y = np.delete(x_last,n_neurons+data_dict['conditioned_neuron_idx'][-1]-1)
    plt.bar(y_first,res_neur[dat_r['conditioned_neuron_idx'][-1],x_last],color=colors) 
    plt.ylim =(-0.2,0.4)
    plt.title('Residual NORM correlation (first vs last trials) r-values for n=' + str(n_neurons) + ' neurons, modulated neuron =' + str(dat_r['conditioned_neuron_idx'][ses]))
    plt.xlabel('Neurons (0-' + str(n_neurons-1) + ')')
    plt.ylabel('r-value (Residual correlation first v last trials)')
    plt.savefig('Residual NORM correlation (first v last trials) r-values mod neuron' + str(dat_r['conditioned_neuron_idx'][ses]) + ' for each neuron session ' + str(ses))
    
    
def cij_coj_NORM_fl():
    
    
#Look at variance between first and last 20 trials per neuron
def var_fl(dat_r,ses):
    n_neurons = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    #n_tri = np.zeros((n_neurons,1))
    for i in range(n_neurons):
        if i == 0:
            n_trials = len(data_dict['f_trialwise_closed_loop'][ses][0,i,:])
        else:
            for m in range(n_trials):
                if np.isnan(data_dict['f_trialwise_closed_loop'][ses][:,i,m]).all() == True:
                    n_trials = m-1
                    break
    last20_all = np.zeros((n_neurons,240,20))
    first20_all = np.zeros((n_neurons,240,20))
    residual_neur_first = np.zeros((n_neurons,240,20))
    residual_neur_last = np.zeros((n_neurons,240,20))
    for q in range(n_neurons):
        first20_all[q,:,:] = dat_r['f_trialwise_closed_loop'][ses][:,q,0:20]
        last20_all[q,:,:] = dat_r['f_trialwise_closed_loop'][ses][:,q,n_trials-21:n_trials-1]
        first20_mean = np.nanmean(first20_all[q,:,:],axis=1)
        last20_mean = np.nanmean(last20_all[q,:,:],axis=1)
        for n in range(20):
             residual_neur_first[q,:,n] = first20_all[q,:,n] - first20_mean
             residual_neur_last[q,:,n] = last20_all[q,:,n] - last20_mean
    residual_neur_firstlast = np.concatenate((residual_neur_first,residual_neur_last))
    return residual_neur_firstlast


#Sliding window of variance changes per neuron (average per 50 trials, 20 trials whatever)

def var_window_neuron(dat_r,ses,window_size):
    n_neurons = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    for i in range(n_neurons):
        if i == 0:
            n_trials = len(data_dict['f_trialwise_closed_loop'][ses][0,i,:])
        else:
            for m in range(n_trials):
                if np.isnan(data_dict['f_trialwise_closed_loop'][ses][:,i,m]).all() == True:
                    n_trials = m-1
                    break
    var_wind_neuron = np.zeros((n_neurons,np.convolve(np.nanvar(dat_r['f_trialwise_closed_loop'][ses][:,0,:],axis=0), np.ones(window_size), 'valid').shape[0]))
    for q in range(n_neurons):
        king = np.convolve(np.nanvar(dat_r['f_trialwise_closed_loop'][ses][:,q,:],axis=0), np.ones(window_size), 'valid') / window_size
        #kk = np.nanmean(king[0:5]) #Normalize variance
        var_wind_neuron[q,:] = king#(king- kk)/kk #Normalize variance
    return var_wind_neuron

#now take the variance across the different time points of the trial
#or should I just take the contribution of each time point in trial of variance (ratio)
def var_window_trial(dat_r,ses,window_size,rm=1):
    n_neurons = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    n_trials = len(dat_r['f_trialwise_closed_loop'][ses][0,0,:])
    var_wind_trial = np.zeros((n_neurons,dat_r['f_trialwise_closed_loop'][ses].shape[0],n_trials))
    for i in range(n_neurons):
        for k in range(dat_r['f_trialwise_closed_loop'][ses].shape[0]):
            x_he = pd.Series(dat_r['f_trialwise_closed_loop'][ses][k,i,:])
            king = x_he.rolling(window_size).var(ddof=0).to_numpy()
            if rm != 1:
                kk = np.nanmean(king[window_size:window_size+5])
                var_wind_trial[i,k,:] = (king- kk)/kk
            else:
                var_wind_trial[i,k,:] = king
    return var_wind_trial


plt.plot(np.nanmean(var_w_neur, axis=0))
plt.title('Moving average of normalized variance across time, mod neuron =68')
plt.xlabel('Trials')
plt.ylabel('Normalized variance units')
plt.savefig('Moving average (n=20 trials) of normalized variance across time, mod neuron =68 of session -1',bbox_inches='tight')

plt.title('Moving average of normalized variance, all neurons, session = -1')
plt.xlabel('Trials')
plt.ylabel('Variance')
plt.savefig('Moving average (n=20 trials) of normalized variance, all neurons, session = -1',bbox_inches='tight')

for i in range(240):
    plt.plot(var_time[68,i])
    plt.title('Moving average of variance across time, mod neuron =68')
    plt.xlabel('Trials')
    plt.ylabel('Variance')
    plt.savefig('Moving average (n=20 trials) of variance across time, mod neuron =68 of session -1',bbox_inches='tight')

#removed the first 39 frames not for planning
for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath); 
        for i in range(len(data_dict['session_dates'])):
            norm_r = normalize_calcium_signal(data_dict)
            var_time_norm = var_window_trial(norm_r,i,20)
            cond_neur = data_dict['conditioned_neuron_idx'][i]
            """
        #error bars
            ci = 1.96 * np.nanstd(var_time_norm[cond_neur,:,:],axis=0)/np.sqrt(var_time_norm.shape[2])
            plt.figure()
            plt.plot(np.nanmean(var_time_norm[cond_neur,:,:],axis=0),'b',label='cond neuron = ' +str(cond_neur))
            plt.fill_between(np.arange(0,var_time_norm.shape[2]),(np.nanmean(var_time_norm[cond_neur,:,:],axis=0)-ci), (np.nanmean(var_time_norm[cond_neur,:,:],axis=0)+ci), color='b', alpha=0.2)
            
            plt.title('REAL df/f Moving average of normalized variance across time, session = '+ str(i) +', mod neuron = ' + str(cond_neur) + ' with 95% CI')
            plt.xlabel('Trials')
            plt.ylabel('Normalized variance units')
            plt.savefig('ERROR BAR Trial time only REAL df-f, Moving average (n=20 trials) of normalized variance across time, mod neuron = ' + str(cond_neur) + ' of session ' + str(i) ,bbox_inches='tight')
            """
#this runs for cond_neurons and variance with new baseline subtraction and normalization (df/f using first 39 frames as baseline)
#done with vairance for onyl conditioned neuron
#now do all neurons
            plt.figure()
            mean_var = np.zeros((var_time_norm.shape[0],var_time_norm.shape[2]))
            #neuron first then time
            """
            for i in range(var_time_norm.shape[0]):
                mean_var[i,:] = np.nanmean(var_time_norm[i,:,:],axis=0)
            ci_m = 1.96 * np.nanstd(mean_var,axis=0)/np.sqrt(var_time_norm.shape[2])
            plt.plot(np.nanmean(mean_var,axis=0),'r',label='all neurons')
            plt.fill_between(np.arange(0,var_time_norm.shape[2]),(np.nanmean(mean_var,axis=0)-ci_m), (np.nanmean(mean_var,axis=0)+ci_m), color='r', alpha=0.2)
            """
            for j in range(var_time_norm.shape[0]):
                mean_var[j,:] = np.nanmean(var_time_norm[j,:,:],axis=0)
            ci_m = 1.96 * np.nanstd(mean_var,axis=0)/np.sqrt(var_time_norm.shape[2])
            plt.plot(np.nanmean(mean_var,axis=0),'r',label='all neurons')
            plt.fill_between(np.arange(0,var_time_norm.shape[2]),(np.nanmean(mean_var,axis=0)-ci_m), (np.nanmean(mean_var,axis=0)+ci_m), color='r', alpha=0.2)
            
            ci = 1.96 * np.nanstd(var_time_norm[cond_neur,:,:],axis=0)/np.sqrt(var_time_norm.shape[2])
            
            plt.plot(np.nanmean(var_time_norm[cond_neur,:,:],axis=0),'b',label='cond neuron = ' +str(cond_neur))
            plt.fill_between(np.arange(0,var_time_norm.shape[2]),(np.nanmean(var_time_norm[cond_neur,:,:],axis=0)-ci), (np.nanmean(var_time_norm[cond_neur,:,:],axis=0)+ci), color='b', alpha=0.2)
            
            plt.title('Moving average of normalized variance across time for all neurons, session = '+ str(i) +', mod neuron = ' + str(cond_neur) + ' with 95% CI')
            plt.xlabel('Trials')
            plt.ylabel('Variance')
            plt.legend()
            plt.savefig('ERROR BAR Trial only Moving average (n=20 trials) of variance all neurons of session '+ str(i) +' mod neuron = ' + str(cond_neur) ,bbox_inches='tight')
#do averaging neuron first and then average per trial, or averaging trials and then neurons first


#take mean of all neurons and calculate confidence interval of 
mean_var = np.zeros((n_neurons,135))
for i in range(219):
    mean_var[i,:] = np.nanmean(var_time[i,:,:],axis=0)
ci = 1.96 * np.nanstd(mean_var,axis=0)/np.sqrt(135)
plt.plot(np.nanmean(mean_var,axis=0))
plt.fill_between(np.arange(0,135),(np.nanmean(mean_var,axis=0)-ci), (np.nanmean(mean_var,axis=0)+ci), color='b', alpha=0.2)
plt.title('Moving (n=20) average of variance all neurons with 95% CI')
plt.xlabel('Trials')
plt.ylabel('Variance')
plt.savefig('ERROR BAR Moving average (n=20 trials) of variance all neurons of session -1',bbox_inches='tight')

#take mean of all neurons and calculate confidence interval
mean_var_norm = np.zeros((n_neurons,135))
for i in range(219):
    mean_var_norm[i,:] = np.nanmean(var_time_norm[i,:,:],axis=0)
ci = 1.96 * np.nanstd(mean_var_norm,axis=0)/np.sqrt(135)
plt.plot(np.nanmean(mean_var_norm,axis=0))
plt.fill_between(np.arange(0,135),(np.nanmean(mean_var_norm,axis=0)-ci), (np.nanmean(mean_var_norm,axis=0)+ci), color='b', alpha=0.2)
plt.title('Moving (n=20) average of normalized variance all neurons with 95% CI')
plt.xlabel('Trials')
plt.ylabel('Normalized variance units')
plt.savefig('ERROR BAR Moving average (n=20 trials) of normalized variance units all neurons of session -1',bbox_inches='tight')
    
#does it make sense to normalize each trial first, then collect the variance? or after I collect the variance?
def var_window_trial_NORM(dat_r,ses,window_size,rm=1):
    n_neurons = len(dat_r['dff_sessionwise_closed_loop'][ses][0,:])
    n_trials = len(data_dict['f_trialwise_closed_loop'][ses][0,0,:])
    var_wind_trial = np.zeros((n_neurons,240,n_trials))
    for i in range(n_neurons):
        for k in range(240):
            king = pd.Series(dat_r['f_trialwise_closed_loop'][ses][k,i,:])
            kk = np.nanmean(king)
            x_he = (king- kk)/kk
            var_wind_trial[i,k,:] = x_he.rolling(window_size).var(ddof=0).to_numpy()
    return var_wind_trial


#take mean of all neurons and calculate confidence interval
mean_var_norm = np.zeros((n_neurons,135))
for i in range(219):
    mean_var_norm[i,:] = np.nanmean(var_wind_trial[i,:,:],axis=0)
ci = 1.96 * np.nanstd(mean_var_norm,axis=0)/np.sqrt(135)
plt.plot(np.nanmean(mean_var_norm,axis=0))
plt.fill_between(np.arange(0,135),(np.nanmean(mean_var_norm,axis=0)-ci), (np.nanmean(mean_var_norm,axis=0)+ci), color='b', alpha=0.2)
plt.title('Moving (n= NORM 20) average of normalized variance all neurons with 95% CI')
plt.xlabel('Trials')
plt.ylabel('Normalized variance units')
plt.savefig('ERROR BAR Moving average (n=20 NORM trials) of normalized variance units all neurons of session -1',bbox_inches='tight')


#do it for all sessions:
import os
for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        for j in range(data_dict['n_days']):
            plt.figure()
            var_m = var_window_trial(data_dict,j,20)    
            n_trials = var_m.shape[2]
            ci = 1.96 * np.nanstd(var_m[data_dict['conditioned_neuron_idx'][j],:,:],axis=0)/np.sqrt(n_trials) 
            plt.plot(np.nanmean(var_m[data_dict['conditioned_neuron_idx'][j],:,:],axis=0))
            plt.fill_between(np.arange(0,n_trials),(np.nanmean(var_m[data_dict['conditioned_neuron_idx'][j],:,:],axis=0)-ci), (np.nanmean(var_m[data_dict['conditioned_neuron_idx'][j],:,:],axis=0)+ci), color='b', alpha=0.2)
            plt.title('Moving average (n=20 trials) of variance, session = '+ str(j) +' mod neuron = '+ str(data_dict['conditioned_neuron_idx'][j]) +' with 95% CI')
            plt.xlabel('Trials')
            plt.ylabel('Variance')
            plt.savefig('ERROR BAR Moving average (n=20 trials) of variance, session = '+ str(j) +' mod neuron = '+ str(data_dict['conditioned_neuron_idx'][j]) +' with 95% CI',bbox_inches='tight')
            
            
#should I quantify the average signal with error bars?
import os
for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        for j in range(data_dict['n_days']):
            plt.figure()
            
            n_trials = data_dict['f_trialwise_closed_loop'][j][:,data_dict['conditioned_neuron_idx'][j],:].shape[1]
            ci = 1.96 * np.nanstd(data_dict['f_trialwise_closed_loop'][j][:,data_dict['conditioned_neuron_idx'][j],:],axis=1)/np.sqrt(240) 
            plt.plot(tr_mean)
            plt.fill_between(np.arange(0,240),(tr_mean-ci), (tr_mean+ci), color='b', alpha=0.2)
            plt.title('Average fluorescence trace, session = '+ str(j) +' mod neuron = '+ str(data_dict['conditioned_neuron_idx'][j]) +' with 95% CI')
            plt.xlabel('Time')
            plt.ylabel('Raw fluorescence signal')
            plt.savefig('ERROR BAR Average fluorescence trace, session = '+ str(j) +' mod neuron = '+ str(data_dict['conditioned_neuron_idx'][j]) +' with 95% CI')
            
            
#realized that they take baseline in a different way. Try to use it? 
#Another option may be to remove the -2.5 to -0.001 part of the trial


#Noise correlations = trial to trial variability of single responses of the neuron
#Signal correlactions = correlation of the average response of the neuron

#noise correlation
#1 do all time points then average r^2 value
#2 average entire signal per trial then take the aerage
def noise_correlation_time_averaged(dat_r,ses,rm=1):
    nor_r = normalize_calcium_signal(dat_r)
    n_neurons = nor_r['f_trialwise_closed_loop'][ses].shape[1]
    expt = nor_r['f_trialwise_closed_loop'][ses]
    #expt = nor_r[~np.isnan(a).any(axis=2)]
    t_points = expt.shape[0]
    noise_corr_neur = np.zeros((n_neurons,t_points))
    cond_neur= dat_r['conditioned_neuron_idx'][ses]
    #for i in range(n_neurons):
    for j in range(n_neurons):
        for k in range(t_points):
            n1_trials = expt[k,cond_neur,:].flatten()
            n1=ma.masked_invalid(n1_trials)
            n2_trials = expt[k,j,:].flatten()
            n2=ma.masked_invalid(n2_trials)
            msk = (~n1.mask & ~n2.mask)
            noise_corr_neur[j,k] = stats.pearsonr(n1_trials[msk],n2_trials[msk])[0]
            #noise_corr_neur[cond_neur,j,k] = 0 #signal.correlate(expt[k,cond_neur,:],expt[k,j,:])
    y_first = np.arange(0,n_neurons)
    no_mod_y = np.delete(y_first,cond_neur)
    #if rm == 1: #average r^2 value for each time point
    r_avg = np.nanmean(noise_corr_neur,axis=1)
    plt.figure()
    colors = ["r" for q in range(n_neurons-1)]
    plt.bar(no_mod_y,r_avg[no_mod_y],color=colors) 
    plt.ylim =(-0.2,0.4)
    plt.title('Time-averaged noise correlation (all trials) r-values for session '+ str(ses) +', n=' + str(n_neurons) + ' neurons, modulated neuron =' + str(cond_neur))
    plt.xlabel('Neurons (0-' + str(n_neurons-1) + ')')
    plt.ylabel('r-value (Averaged noise correlation all trials)')
    plt.savefig('Time-averaged noise correlation (all trials) r-values for session '+ str(ses) +', n=' + str(n_neurons) + ' neurons, modulated neuron =' + str(cond_neur),bbox_inches='tight')
    return plt.show(), np.save('Noise_correlations_session_'+ str(ses) +'_modneur_'+ str(cond_neur) +'.npy',noise_corr_neur)


#do noise correlations for all
#should I quantify the average signal with error bars?
import os
for k in range(9):
    if k == 2:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = read_multisession_mat_NEW(fullpath);
    elif k == 7 or k == 3 or k == 5:
        skip=0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
    norm_r = normalize_calcium_signal(data_dict)
    #norm_r_nonan = remove_calcium_nans(norm_r)
    """
    for j in range(data_dict['n_days']):
        if type(data_dict['f_trialwise_closed_loop'][j]) is np.ndarray:
            #noise_correlation_time_averaged(data_dict,j)
            plt.figure()
            norm_r_avg = np.nanmean(norm_r['f_trialwise_closed_loop'][j],axis=2)
            norm_r_avg_new = norm_r_avg
    
            if np.where(norm_r_avg < -5)[0].size != 0:
                xn5 = np.where(norm_r_avg < -5)
                norm_r_avg_new = np.delete(norm_r_avg,np.unique(xn5[1],return_counts=True)[0][-1],axis=1)
            elif np.where(norm_r_avg > 5)[0].size != 0:
                xp5 = np.where(norm_r_avg > 5)
                norm_r_avg_new = np.delete(norm_r_avg,np.unique(xp5[1],return_counts=True)[0][-1],axis=1)
    
            time_t = np.linspace(0,10,num=norm_r_avg_new[39:,:].shape[0])
            for i in range(norm_r_avg_new.shape[1]):                
                if i == data_dict['conditioned_neuron_idx'][j]:
                    plt.plot(time_t, norm_r_avg_new[39:,i],'b')
                else:
                    plt.plot(time_t, norm_r_avg_new[39:,i],'r')
            plt.plot(time_t, norm_r_avg_new[39:,data_dict['conditioned_neuron_idx'][j]],'b')
            plt.xlabel('Time points')
            plt.ylabel('ΔF/F')
            plt.title('NEW Bad neuron removed Trial-averaged values for n=' + str(norm_r_avg_new.shape[1]) + ' neurons, blue = neuron ' + str(data_dict['conditioned_neuron_idx'][j]))
            os.chdir(basepath + 'behavior')
            plt.savefig('NEW Bad neuron Neurons trial averaged traces, session' + str(j) + ' dF-f', bbox_inches='tight')
    """
    """
    for j in range(data_dict['n_days']):
         if type(data_dict['f_trialwise_closed_loop'][j]) is np.ndarray:
            plt.figure()
            var_m = var_window_trial(norm_r,j,20)    
            n_trials = var_m.shape[2]
            ci = 1.96 * np.nanstd(var_m[norm_r['conditioned_neuron_idx'][j],:,:],axis=0)/np.sqrt(n_trials) 
            plt.plot(np.nanmean(var_m[norm_r['conditioned_neuron_idx'][j],:,:],axis=0))
            plt.fill_between(np.arange(0,n_trials),(np.nanmean(var_m[norm_r['conditioned_neuron_idx'][j],:,:],axis=0)-ci), (np.nanmean(var_m[norm_r['conditioned_neuron_idx'][j],:,:],axis=0)+ci), color='b', alpha=0.2)
            plt.title('Moving average (n=20 trials) of variance, session = '+ str(j) +' mod neuron = '+ str(norm_r['conditioned_neuron_idx'][j]) +' with 95% CI')
            plt.xlabel('Trials')
            plt.ylabel('Variance')
            os.chdir(basepath + 'behavior')
            plt.savefig('NEW ERROR BAR Moving average (n=20 trials) of variance, session = '+ str(j) +' mod neuron = '+ str(data_dict['conditioned_neuron_idx'][j]) +' with 95% CI',bbox_inches='tight')
    """
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)
#imputation of calcium signal NaNs with k-nearest neighbors
    impute_r = norm_r
    n_session_game = np.zeros((data_dict['n_days']),dtype=object)
    for i in range(data_dict['n_days']):
        if type(impute_r['f_trialwise_closed_loop'][i]) is np.ndarray:
            n_trials = len(norm_r['f_trialwise_closed_loop'][i][0,0,:])
            n_neurons = len(norm_r['f_trialwise_closed_loop'][i][0,:,0])
            print(str(i) + 'session')
            n_neuron_game = np.zeros((n_neurons),dtype=object)
            norm_r['f_trialwise_closed_loop'][i][np.isinf(impute_r['f_trialwise_closed_loop'][i][:,:,:])] = np.nan
            for j in range(n_neurons):
                #impute_r['f_trialwise_closed_loop'][i][np.isinf(impute_r['f_trialwise_closed_loop'][i][:,:,j])] = np.nan
                #df_neurons = pd.DataFrame(norm_r['f_trialwise_closed_loop'][i][:,j,:])
                #df_neurons.replace([np.inf, -np.inf], np.nan, inplace=True)
                #df_neurons.fillna(method ='ffill')
                #impute_r['f_trialwise_closed_loop'][i][:,j,:] = df_neurons.to_numpy()
                #print(str(j) + 'trial')
                impute_r['f_trialwise_closed_loop'][i][:,j,:] = imputer.fit_transform(norm_r['f_trialwise_closed_loop'][i][:,j,:]); 
            #pca=PCA(n_components=40,svd_solver='full')
            n_trials = len(impute_r['f_trialwise_closed_loop'][i][0,0,:])
            pca_comp = np.zeros(n_trials,dtype=object)
            for w in range(n_trials):
                print(w)
                """
                #n1_trials = norm_r['f_trialwise_closed_loop'][0][:,:,i].flatten()
                pca_mask = ma.masked_invalid(norm_r['f_trialwise_closed_loop'][0][:,:,i])
                pca.fit(pca_mask)
                pca_comp[i] = pca.explained_variance_ratio_
                """
                pca=PCA(svd_solver='full')
                #pca.fit(n_session_game[0][i])
                pca.fit(impute_r['f_trialwise_closed_loop'][i][:,:,w])
                pca_comp[w] = pca.explained_variance_ratio_
                
            pc_num_90 = np.zeros(n_trials)
            for z in range(n_trials):
                """
                if np.where(np.cumsum(pca_comp[z]) >= 0.80)[0].size == 0:
                    
                    pc_num_90[z] = 40
                else:
                """
                pc_num_90[z] = np.min(np.where(np.cumsum(pca_comp[z]) >= 0.80))
            plt.figure()
            plt.plot(pc_num_90)
            #n_session_game[i] = n_neuron_game 
            #plt.plot(np.nanmean(var_m[norm_r['conditioned_neuron_idx'][j],:,:],axis=0))
            if max(pc_num_90) > 30:
                plt.yticks(np.arange(0, max(pc_num_90)+1, 5.0))    
            else:
                plt.yticks(np.arange(0, max(pc_num_90)+1, 1.0))
            #plt.fill_between(np.arange(0,n_trials),(np.nanmean(var_m[norm_r['conditioned_neuron_idx'][j],:,:],axis=0)-ci), (np.nanmean(var_m[norm_r['conditioned_neuron_idx'][j],:,:],axis=0)+ci), color='b', alpha=0.2)
            plt.title('Principal components required across trials, all neurons, session = ' + str(i) + ' mod neuron = ' + str(norm_r['conditioned_neuron_idx'][i]) )
            plt.xlabel('Trials')
            plt.ylabel('PCs for 80% of variance')
            os.chdir(basepath + 'behavior')
            plt.savefig('Principal components required for 90% of variance across trials, all neurons, session = '+ str(i) +' mod neuron = '+ str(data_dict['conditioned_neuron_idx'][i]),bbox_inches='tight')



# use this
#after you use PC's, maybe you can explain which ones are correct trials vs incorrect trials


