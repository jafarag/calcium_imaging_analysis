# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 09:53:17 2022

@author: User
"""

"""
By separating trials via trial outcome (lick reward or no lick reward), it would be good to separate whether there are any changes in neural activity, 
correlation matrix structure, and dimensionality of neural data when seaprated by trial outcome. I also take a look at whether we can predict the 
signal of the conditioned neuron from the other neurons (usign linear regression) and how that compares across different trial outcomes.

"""


#import BCI_analysis.BCI_analysis as bci
#data_dict = bci.io_matlab.read_multisession_mat('C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-002/BCI_data/BCI22_030222v8.mat');

import os
os.chdir("C:/Users/User")
import BCI_analysis.BCI_analysis as bci
import scipy 
import numpy as np
import matplotlib.pyplot as plt

for k in range(8):
    if k == 2 or k == 3 or k == 5 or k == 7:
        skip = 0;
    else:
        basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-00' + str(k+1) + '/BCI_data/'
        os.chdir(basepath)
        dirpath = os.listdir(basepath)
        fullpath = basepath + dirpath[0]        
        data_dict = bci.io_matlab.read_multisession_mat(fullpath);
        for j in range(len(data_dict['session_dates'])):
            print('Mouse'+ str(k)+'Session ' + str(j) + ': ' + str(data_dict['session_dates'][j]))



basepath = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-005/BCI_data/'
os.chdir(basepath)
dirpath = os.listdir(basepath)
fullpath = basepath + dirpath[0]        
data_dict = bci.io_matlab.read_multisession_mat(fullpath);
beh = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-001/BCI_data/behavior/BCI_11'
os.chdir(beh)
dirpath = os.listdir(basepath)
#fullpath = beh + '070721-bpod_zaber.npy'
behmat = scipy.io.loadmat('070721-bpod_zaber.mat')
#trial hit gives you trials that were successfully rewarded
correct = np.where(behmat['trial_hit']==1)[1]
false = np.where(behmat['trial_hit']==0)[1]
data_correct = data_dict['f_trialwise_closed_loop'][0][:,:,correct]
data_false = data_dict['f_trialwise_closed_loop'][0][:,:,false]
data_correct = data_correct[39:,:,:]
data_false = data_false[39:,:,:]
"""
#second 070721
beh13 = 'C:/Users/User/Downloads/Mouse_BCI/BCI_data-20220331T182839Z-001/BCI_data/behavior/BCI_13'
os.chdir(beh13)
dirpath13 = os.listdir(beh13)
#fullpath = beh + '070721-bpod_zaber.npy'
behmat13 = scipy.io.loadmat('070721-bpod_zaber.mat')
#trial hit gives you trials that were successfully rewarded
correct13 = np.where(behmat13['trial_hit']==1)[1]
false13 = np.where(behmat13['trial_hit']==0)[1]
data_correct13 = data_dict['f_trialwise_closed_loop'][0][:,:,correct13]
data_false13 = data_dict['f_trialwise_closed_loop'][0][:,:,false13]
"""

condneur = data_dict['conditioned_neuron_idx'][0]

#idea now is to plot trial-averaged trace (raw) between reward and failure plots
correct_mean = np.nanmean(data_correct[:,condneur,:],axis=1)
false_mean = np.nanmean(data_false[:,condneur,:],axis=1)
#n_trials = data_dict['f_trialwise_closed_loop'][j][:condneur,:].shape[1]
correct_ci = 1.96 * np.nanstd(data_correct[:,condneur,:],axis=1)/np.sqrt(len(correct))
false_ci = 1.96 * np.nanstd(data_false[:,condneur,:],axis=1)/np.sqrt(len(false)) 
plt.figure()
plt.plot(correct_mean,color='b',label='Rewarded trials')
plt.fill_between(np.arange(0,201),(correct_mean-correct_ci), (correct_mean+correct_ci), color='b', alpha=0.2)
plt.plot(false_mean,color='r',label='Non-rewarded trials')
plt.fill_between(np.arange(0,201),(false_mean-false_ci), (false_mean+false_ci), color='r', alpha=0.2)
plt.plot(np.nanmean(data_dict['f_trialwise_closed_loop'][0][39:,condneur,:],axis=1),color='g')
plt.title('Average fluorescence trace reward (b) vs none (r) at trial start, session = '+ str(j) +' mod neuron = '+ str(condneur) +' with 95% CI')
plt.xlabel('Time')
plt.ylabel('Raw fluorescence signal')
plt.savefig('At trial start: Reward vs none ERROR BAR Average fluorescence trace, session = '+ str(j) +' mod neuron = '+ str(condneur) +' with 95% CI')

 
#If the 
data_c = data_dict['f_trialwise_closed_loop'][0][:,:,correct]
correct_restneur_c = np.delete(data_c[:39,:,:],6,1)
correct_condneur_c = data_c[:39,condneur,:]
regr = LinearRegression().fit(correct_restneur_c[:,:,0], correct_condneur_c[:,0])
infin = []
for i in range(92):
    if np.all(np.isfinite(correct_restneur[:,:,i])) != True:
        print(i)
        infin.append(i)
    elif np.all(np.isfinite(correct_condneur[:,i])) != True:
        print(i + "new")
        infin.append(i)
det_score = np.zeros(78)
all_t = list(np.arange(92))
finite_trials = np.array([i for i in all_t if i not in infin])
for i in range(len(finite_trials)):
    if np.all(np.isfinite(correct_restneur[:,:,i])) != True:
        print(i)
        infin.append(i)
    elif np.all(np.isfinite(correct_condneur[:,i])) != True:
        print(i + "new")
        infin.append(i)



for i in range(78):
    regr = LinearRegression().fit(correct_restneur_c[:,:,finite_trials[0]], correct_condneur_c[:,finite_trials[0]])
    det_score[i] = regr.score(correct_restneur[:,:,finite_trials[i]], correct_condneur[:,finite_trials[i]])
plt.plot(det_score)
plt.title('scoring from same trial, normalized dF/f')
#use previous trial to predict the next one
det_score = np.zeros(78)
new_det_score = np.zeros(78)
for i in range(1,78):
    regr = LinearRegression().fit(correct_restneur[:,:,finite_trials[i-1]], correct_condneur[:,finite_trials[i-1]])
#add test and metric
    pred_condneur = regr.predict(correct_restneur[:,:,finite_trials[i]])
    new_det_score[i] = r2_score(correct_condneur[:,finite_trials[i]],pred_condneur)
#regular
    det_score[i] = regr.score(correct_restneur[:,:,finite_trials[i]], correct_condneur[:,finite_trials[i]])
plt.plot(det_score)
plt.title('scoring from previous trial, normalized dF/f')   

theta_ols = np.linalg.inv(np.matmul(correct_restneur_2.T,correct_restneur_2)) @ correct_restneur_2.T @ correct_condneur
Yhat =  correct_restneur_2 @ theta_ols

#result if you add pred function? no difference from normal scoring 
#next step: use different regression to predict it 


#standardize before regression
stand_det_score = np.zeros(78)
for i in range(1,78):
    regr = LinearRegression().fit(xa_correct_restneur[:,:,finite_trials[i-1]], xa_correct_condneur[:,finite_trials[i-1]])
#add test and metric
    xa_pred_condneur = regr.predict(xa_correct_restneur[:,:,finite_trials[i]])
    stand_det_score[i] = r2_score(xa_correct_condneur[:,finite_trials[i]],xa_pred_condneur)
    
plt.plot(stand_det_score)
plt.title('standardized scoring from previous trial, normalized dF/f') 
#result: still negative r^2 score, but not as negative as others
#reshape data? add more trials

xd = correct_restneur.reshape((201*97,92))
resh_det_score = np.zeros(78)

regr = LinearRegression().fit(xd[:,finite_trials[0:5]].T, correct_condneur[:,finite_trials[0:5]].T)
for i in range(5,78):    
#add test and metric
    dd = xd[:,finite_trials[i]].reshape(-1,1)
    pred_condneur = regr.predict(dd.T)
    resh_det_score[i] = r2_score(correct_condneur[:,finite_trials[i]],pred_condneur.T)
plt.plot(resh_det_score)
plt.title('train on prev 5 scoring from previous trial (now early 5 trials), normalized dF/f', bbox_inches='tight')   
#result: slightly better model fits, but still not great
#if you only train at beginning, then you get good result
#if you standardize, then you get a better model fit
#next: add more trials? Better model score but also worse


#shuffle samples
#xd = correct_restneur.reshape((201*97,92))
shuf_det_score = np.zeros(78)
xds = xd[:,finite_trials]
correct_condneur_s = correct_condneur[:,finite_trials]
xd_shuffled, correct_condneur_shuffled = sklearn.utils.shuffle(xds.T,correct_condneur_s.T)
regr = LinearRegression().fit(xd_shuffled[finite_trials[0:5],:], correct_condneur_shuffled[finite_trials[0:5],:])
for i in range(5,78):    
#add test and metric
    dd = xds[:,i].reshape(-1,1)
    pred_condneur = regr.predict(dd.T)
    shuf_det_score[i] = r2_score(correct_condneur[:,finite_trials[i]],pred_condneur.T)
plt.plot(shuf_det_score)
plt.title('SHUFFLED train on prev 5 scoring from previous trial (now early 5 trials), normalized dF/f', bbox_inches='tight')

#you need to normalize the calcium signal using baseline of pre trial activity
def normalize_calcium_signal(data_r,delete_all=0,baseline_start_time = -2.5,baseline_end_time=0,start_time=-2.5,end_time=10):
    """Input: data_r: data dictionary of successful or unsuccessful trials
    Output: norma_r: data dictionary of successful or unsuccessful trials normalized by first 39 trials
    """
    norma_r = data_r
    norma_ra = norma_r
    for j in range(data_r['n_days']):
        if type(data_r['f_trialwise_closed_loop'][j]) is np.ndarray:
            n_neurons = len(data_r['f_trialwise_closed_loop'][j][0,:])
            time_today=np.transpose(norma_r['time_from_trial_start'][j]).flatten()
            idx_today = (time_today>start_time) & (time_today<end_time)
            time_today = time_today[idx_today]
            norma_r['f_trialwise_closed_loop'][j] = data_r['f_trialwise_closed_loop'][j][:-6,:,:]
            norma_r['f_trialwise_closed_loop'][j][np.isinf(norma_r['f_trialwise_closed_loop'][j][:,:,:])] = np.nan
            #baseline_idx_today = (time_today>baseline_start_time) & (time_today<baseline_end_time)
            """
            for i in range(n_neurons):
                n_trials = len(data_r['f_trialwise_closed_loop'][j][0,i,:])
                neur_trials = []
                neur_neur = []
                if np.isnan(norma_r['f_trialwise_closed_loop'][j][:,i,:]).all() == True:
                    neur_neur.append(i)
            if bool(neur_neur) != False:
                norma_ra['f_trialwise_closed_loop'][j] = np.delete(norma_r['f_trialwise_closed_loop'][j],neur_neur,axis=1)        
            for m in range(n_trials):
                if np.isnan(norma_r['f_trialwise_closed_loop'][j][:,:,m]).all() == True:
                    neur_trials.append(m)
                    #n_trials = m-1
                    #break
            if bool(neur_trials) != False:
                norma_ra['f_trialwise_closed_loop'][j] = np.delete(norma_r['f_trialwise_closed_loop'][j],neur_trials,axis=2)
            else:
                print('no')
            """
            bad_trials = []
            for i in range(data_dict['f_trialwise_closed_loop'][j].shape[2]):
                nan_ratio = np.isnan(data_dict['f_trialwise_closed_loop'][j][:,:,i]).sum()/(data_dict['f_trialwise_closed_loop'][j][:,:,i].shape[0]*data_dict['f_trialwise_closed_loop'][j][:,:,i].shape[1])
                if nan_ratio >= 0.15:
                    bad_trials.append(i)
            norma_ra['f_trialwise_closed_loop'][j] = np.delete(norma_r['f_trialwise_closed_loop'][j],bad_trials,axis=2)
            print(bad_trials)
            print('session' + str(j))
            n_trials = len(norma_ra['f_trialwise_closed_loop'][j][0,0,:])    
            for g in range(n_neurons):
                for k in range(n_trials):
                    if np.isnan(norma_ra['f_trialwise_closed_loop'][j][:39,g,k]).all() == True:
                        norma_ra['f_trialwise_closed_loop'][j][:,g,k]= (norma_r['f_trialwise_closed_loop'][j][:,g,k] - np.nanmean(norma_r['f_trialwise_closed_loop'][j][40:45,g,k],0)) / np.nanmean(norma_r['f_trialwise_closed_loop'][j][40:45,g,k],0)
                    else:
                        norma_ra['f_trialwise_closed_loop'][j][:,g,k]= (norma_r['f_trialwise_closed_loop'][j][:,g,k] - np.nanmean(norma_r['f_trialwise_closed_loop'][j][:39,g,k],0)) / np.nanmean(norma_r['f_trialwise_closed_loop'][j][:39,g,k],0)
            norma_ra['f_trialwise_closed_loop'][j][np.isinf(norma_ra['f_trialwise_closed_loop'][j][:,:,:])] = np.nan
            bad2_trials = []
            for i in range(norma_ra['f_trialwise_closed_loop'][j].shape[2]):
                nan_ratio = np.isnan(data_dict['f_trialwise_closed_loop'][j][:,:,i]).sum()/(data_dict['f_trialwise_closed_loop'][j][:,:,i].shape[0]*data_dict['f_trialwise_closed_loop'][j][:,:,i].shape[1])
                if nan_ratio >= 0.15:
                    bad2_trials.append(i)
            norma_ra['f_trialwise_closed_loop'][j] = np.delete(norma_r['f_trialwise_closed_loop'][j],bad2_trials,axis=2)
    return norma_ra
                

#should I dispose of nans in trials or in individual time points to comapre

#you need to normalize the calcium signal using baseline of pre trial activity
def remove_calcium_nans(data_r,baseline_start_time = -2.5,baseline_end_time=0,start_time=-2.5,end_time=10):
    """Input: data_r: data dictionary of successful or unsuccessful trials
    Output: norma_r: data dictionary of successful or unsuccessful trials normalized by first 39 trials
    """
    norma_r = data_r
    for j in range(data_r['n_days']):
            n_neurons = len(data_r['dff_sessionwise_closed_loop'][j][0,:])
            time_today=np.transpose(norma_r['time_from_trial_start'][j]).flatten()
            idx_today = (time_today>start_time) & (time_today<end_time)
            time_today = time_today[idx_today]
            norma_r['f_trialwise_closed_loop'][j] = data_r['f_trialwise_closed_loop'][j][:-6,:,:]
            n_trials = len(data_r['f_trialwise_closed_loop'][j][0,0,:])
            #baseline_idx_today = (time_today>baseline_start_time) & (time_today<baseline_end_time)
            neur_trials = [] #np.zeros((n_neurons)),dtype=object)
            neurs = []
            for i in range(n_neurons):
                for m in range(n_trials):                
                    if np.isnan(data_r['f_trialwise_closed_loop'][j][:,i,m]).all() == True:
                        neur_trials.append(m)
            norma_v = np.delete(norma_r['f_trialwise_closed_loop'][j],neur_trials,axis=2)
    return norma_v



"""
Mouse0Session 0: 2321-01-06
Mouse0Session 1: 0521-01-09
Mouse0Session 2: 0621-01-09
Mouse0Session 3: 0721-01-09
Mouse0Session 4: 1021-01-09
Mouse1Session 0: 1121-01-01
Mouse1Session 1: 1321-01-01
Mouse1Session 2: 1521-01-01
Mouse4Session 0: 0721-01-07
Mouse4Session 1: 3121-01-08
Mouse4Session 2: 0521-01-09
Mouse4Session 3: 0621-01-09
Mouse4Session 4: 0721-01-09
Mouse4Session 5: 0821-01-09
Mouse6Session 0: 2321-01-01
Mouse6Session 1: 2421-01-01
"""
