# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:21:08 2022

@author: User
"""
data_dict = bci.io_matlab.read_multisession_mat(fullpath);
norm_r = normalize_calcium_signal(data_dict)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
#imputation of calcium signal NaNs with k-nearest neighbors
impute_r = norm_r
n_session_game = np.zeros((data_dict['n_days']),dtype=object)
for i in range(data_dict['n_days']):
    n_trials = len(norm_r['f_trialwise_closed_loop'][i][0,0,:])
    n_neurons = len(norm_r['f_trialwise_closed_loop'][i][0,:,0])
    print(str(i) + 'session')
    n_neuron_game = np.zeros((n_neurons),dtype=object)
    #impute_r['f_trialwise_closed_loop'][i][np.isinf(impute_r['f_trialwise_closed_loop'][i][:,:,:])] = np.nan
    for j in range(n_trials):
        #impute_r['f_trialwise_closed_loop'][i][np.isinf(impute_r['f_trialwise_closed_loop'][i][:,:,j])] = np.nan
        #df_neurons = pd.DataFrame(norm_r['f_trialwise_closed_loop'][i][:,j,:])
        #df_neurons.replace([np.inf, -np.inf], np.nan, inplace=True)
        #df_neurons.fillna(method ='ffill')
        #impute_r['f_trialwise_closed_loop'][i][:,j,:] = df_neurons.to_numpy()
        print(str(j) + 'trial')
        impute_r['f_trialwise_closed_loop'][i][:,:,j] = imputer.fit_transform(impute_r['f_trialwise_closed_loop'][i][:,:,j]); 
    #n_session_game[i] = n_neuron_game    

#PCA decomposition of normalized signals

from sklearn.decomposition import PCA
pca=PCA(n_components=40,svd_solver='full')
n_trials = len(impute_r['f_trialwise_closed_loop'][0][0,0,:])
pca_comp = np.zeros(n_trials,dtype=object)
for i in range(n_trials):
    print(i)
    """
    #n1_trials = norm_r['f_trialwise_closed_loop'][0][:,:,i].flatten()
    pca_mask = ma.masked_invalid(norm_r['f_trialwise_closed_loop'][0][:,:,i])
    pca.fit(pca_mask)
    pca_comp[i] = pca.explained_variance_ratio_
    """
    #pca.fit(n_session_game[0][i])
    pca.fit(norm_r['f_trialwise_closed_loop'][0][:,:,i])
    pca_comp[i] = pca.explained_variance_ratio_
    
pc_num_90 = np.zeros(n_trials)
for i in range(n_trials):
    k = 0
    q = 0
    for j in range(40):
        k += 1
        q += pca_comp[i][j]
        if q >= 0.9:
            pc_num_90[i] =  k
            break

plt.plot(pc_num_90)

def KNN_imputation(norm_dic):
    
