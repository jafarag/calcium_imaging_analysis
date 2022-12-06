# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:28:29 2022

@author: User
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
            first50_corr_n = noise_corr(first50_all,first50_all)
            last50_corr_n = noise_corr(last50_all,last50_all)
            n27_first = first50_corr_n[27,:]
            n27_last = last50_corr_n[27,:]
            idxlag_p = np.where(n27_last>20)
            idxlag_n = np.where(n27_last<-20)
            #idxlagn100 = np.argwhere(-100< n72_n1 < 0)
            #idxlag100 = np.argwhere(100> n72_n1 > 0)
            n27_last_n = n27_first[idxlag_n]
            n27_last_p = n27_first[idxlag_p]
            idxlagn200 = np.where(n27_first < -100)
            idxlag200 = np.where(n27_first > 100)
            idxlag100 = np.where((n27_first < 100)&(n27_first > 20))
            idxlagn100 = np.where((n27_first > -100)&(n27_first < -20))
            idxlagmid = np.where((20>n27_first)&(n27_first>-20)) 
            y_coord = data_dict['roi_center_y'][j]
            x_coord = data_dict['roi_center_x'][j]
            y_coord_real = np.zeros((len(y_coord)+1))    
            for i in range(len(y_coord)):
                y_coord_real[i] = y_coord[i][1] 
            for h in range(n_neurons):
                if h == data_dict['conditioned_neuron_idx'][j]:
                    plt.scatter(x=y_coord_real[h],y=x_coord[h],c='b', label='conditioned neuron')
                elif np.isin(h,idxlagmid[0]) == True:
                    plt.scatter(x=y_coord_real[h],y=x_coord[h],c='c')#, label='-200<lag<-100')  
                elif np.isin(h,idxlagn100) == True:
                    plt.scatter(x=y_coord_real[h],y=x_coord[h],c='m') #label='-100<lag<0')
                elif np.isin(h,idxlag100) == True:
                    plt.scatter(x=y_coord_real[h],y=x_coord[h],c='r')# label='0>lag>100')
                elif np.isin(h,idxlag200) == True:
                    plt.scatter(x=y_coord_real[h],y=x_coord[h],c='g')#, label='200>lag>100')
                elif np.isin(h,idxlagn200) == True:
                    plt.scatter(x=y_coord_real[h],y=x_coord[h],c='k')#, label='-200<lag<-100')
                else:
                    plt.scatter(x=y_coord_real[h],y=x_coord[h],marker='+',c='k')
                    print(h)
            #sns.heatmap(first50_corr_n,cmap='seismic')
            #plt.plot(first50_all[data_dict['conditioned_neuron_idx'][j],:],'b')
            plt.xlabel('X (μm)')
            plt.ylabel('Y (μm)')
            plt.legend()
            plt.title('Projected first 50 trials lagtimes for n=' + str(n_neurons) + ' neurons, blue = neuron ' + str(data_dict['conditioned_neuron_idx'][j]))
            plt.savefig('Lagtimes in space first 50 trials session ' + str(j) + ' dF-f Noise corr')






n27_first = first50_corr_n[27,:]
n27_last = last50_corr_n[27,:]


idxlag_p = np.where(n27_first>0)
idxlag_n = np.where(n27_first<0)
#idxlagn100 = np.argwhere(-100< n72_n1 < 0)
#idxlag100 = np.argwhere(100> n72_n1 > 0)
idxlagn200 = np.argwhere(n72_first[idxlag_n] < -100)
idxlag200 = np.argwhere(n72_first[idxlag_p] > 100)
idxlag100 = np.argwhere(n72_first[idxlag200] < 100)
idxlagn100 = np.argwhere(n72_first[idxlagn200] > -100)


plt.scatter(y_coord_real,x_coord)
plt.scatter(x=y_coord_real[27],y=x_coord[27],c='r', label='conditioned neuron')
plt.show()

            n27_last_n = n27_first[idxlag_n]
            n27_last_p = n27_first[idxlag_p]
            idxlagn200 = np.where(n27_first < -100)
            idxlag200 = np.where(n27_first > 100)
            idxlag100 = np.where((n27_first < 100)&(n27_first > 20))
            idxlagn100 = np.where((n27_first > -100)&(n27_first < -20))
            idxlagmid = np.where((20>n27_first)&(n27_first>-20))  
