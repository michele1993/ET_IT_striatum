import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os, sys
#from openTSNE import TSNE
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils import get_data
from unsupervised_cortex.preTrain_FF_cortex.IT_NN import IT_NN
from unsupervised_cortex.ValueBased_model.Striatum_lNN import Striatum_lNN



seeds_1 = [62419, 87745, 55327, 31023, 21716]
seeds_2 = [91207, 61790, 12391, 57053, 81513]
seeds = seeds_1 + seeds_2

stde_norm = np.sqrt(len(seeds))

font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s

# Create a figure and a set of subplots
fig, axs = plt.subplots(4, 3, figsize=(7, 6.5),#subplot_kw=dict(projection='3d'),
                        gridspec_kw={'wspace': 0.65, 'hspace': 0.8, 'left': 0.09, 'right': 0.98, 'bottom': 0.1,'top': 0.94})



#file_dir = os.path.dirname(os.path.abspath(__file__))
#file_dir = os.path.join(file_dir,'data')

## ----- Load experimental data ------
mat = scipy.io.loadmat('experimental_data/LickrateData.mat')
lick_rate = np.squeeze(mat['Lickrate'])
Healthy_lick_CSp = lick_rate[0][0].mean(axis=1)
Healthy_lick_CSm = lick_rate[0][1].mean(axis=1)

NoIT_lick_CSp = lick_rate[1][0].mean(axis=1)
NoIT_lick_CSm = lick_rate[1][1].mean(axis=1)

NoET_lick_CSp = lick_rate[2][0].mean(axis=1)
NoET_lick_CSm = lick_rate[2][1].mean(axis=1)

NoET_IT_lick_CSp = lick_rate[3][0].mean(axis=1)
NoET_IT_lick_CSm  = lick_rate[3][1].mean(axis=1)

## normalise lick rate between 0 and 1 by diving by max lick rate across all conditions
## this occured in the no ET CS+ condition
normaliser = 5 # np.max(NoET_lick_CSp) ## norm by 5Hz closest interger to max lick rate

Healthy_lick_CSp /= normaliser
Healthy_lick_CSm /= normaliser

NoIT_lick_CSp /=normaliser
NoIT_lick_CSm /=normaliser

NoET_lick_CSp /=normaliser
NoET_lick_CSm /=normaliser

NoET_IT_lick_CSp /=normaliser
NoET_IT_lick_CSm  /=normaliser



## ----- Load simulated data ---------
dataset_name = 'synthetic_data' #'cifar10'
# Initialise useful lists
Healthy_CSp_val = []
Healthy_CSm_val = []
NoET_CSp_val = []
NoET_CSm_val = []
NoIT_CSp_val = []
NoIT_CSm_val = []
NoET_IT_CSp_val = []
NoET_IT_CSm_val = []
Healthy_noCortex_acc = []
Healthy_cortexDep_acc = []
striatal_acc_NoET = []
striatal_acc_NoIT = []

##  ----- Main analysis based on class comparison: 0 -----
# Initialise all classes comparisons
all_classes_comp = [[0,2],[0,1],[0,4],[0,5]]
class_comp = 0
specific_classes = all_classes_comp[0]
for s in seeds:
    data_1 = f'data/VB_{dataset_name}_LickData_seed_{s}_class_{specific_classes[0]}_vs_{specific_classes[1]}'
    data_2 = data_1 + '_noET_.pt'
    data_3 = data_1 + '_noIT_.pt'
    data_4 = data_1 + '_noET__noIT_.pt'
    data_1 += '.pt'

    data_1 = torch.load(data_1)
    data_2 = torch.load(data_2)
    data_3 = torch.load(data_3)
    data_4 = torch.load(data_4)

    ## Extract values
    Healthy_CSp_val.append(data_1['CSp_rwd'])
    Healthy_CSm_val.append(data_1['CSm_rwd'])

    NoET_CSp_val.append(data_2['CSp_rwd'])
    NoET_CSm_val.append(data_2['CSm_rwd'])

    NoIT_CSp_val.append(data_3['CSp_rwd'])
    NoIT_CSm_val.append(data_3['CSm_rwd'])

    NoET_IT_CSp_val.append(data_4['CSp_rwd'])
    NoET_IT_CSm_val.append(data_4['CSm_rwd'])

    Healthy_noCortex_acc.append(data_1['NoCortex_acc'])
    Healthy_cortexDep_acc.append(data_1['Striatal_acc'])

    striatal_acc_NoET.append(data_2['Striatal_acc'])
    striatal_acc_NoIT.append(data_3['Striatal_acc'])


# convert everything to np.array
#indx_learning_stages = [0, 9, 19, 29, 39, 49]
indx_learning_stages = [0, 7, 15, 23, 31, 39]
Healthy_CSp_val = np.array(Healthy_CSp_val)[:,indx_learning_stages] 
Healthy_CSm_val = np.array(Healthy_CSm_val)[:,indx_learning_stages]
NoET_CSp_val = np.array(NoET_CSp_val)[:,indx_learning_stages]
NoET_CSm_val = np.array(NoET_CSm_val)[:,indx_learning_stages]
NoIT_CSp_val = np.array(NoIT_CSp_val)[:,indx_learning_stages]
NoIT_CSm_val = np.array(NoIT_CSm_val)[:,indx_learning_stages]
NoET_IT_CSp_val = np.array(NoET_IT_CSp_val)[:,indx_learning_stages]
NoET_IT_CSm_val = np.array(NoET_IT_CSm_val)[:,indx_learning_stages]

## Insert baseline at trial zero since saved results from averaged of first 100 trials
Healthy_CSp_val[:,0] = 0
Healthy_CSm_val[:,0] = 0
NoET_CSp_val[:,0] = 0
NoET_CSm_val[:,0] = 0
NoIT_CSp_val[:,0] = 0
NoIT_CSm_val[:,0] = 0
NoET_IT_CSp_val[:,0] = 0
NoET_IT_CSm_val[:,0] = 0


Healthy_noCortex_acc = np.array(Healthy_noCortex_acc) * 100
Healthy_cortexDep_acc = np.array(Healthy_cortexDep_acc) * 100
striatal_acc_NoET = np.array(striatal_acc_NoET) * 100
striatal_acc_NoIT = np.array(striatal_acc_NoIT) * 100

## Compute mean and std
# Healthy
Healthy_CSp_val_mean = np.mean(Healthy_CSp_val, axis=0) 
Healthy_CSp_val_std = np.std(Healthy_CSp_val, axis=0)/ stde_norm 
Healthy_CSm_val_mean = np.mean(Healthy_CSm_val, axis=0) 
Healthy_CSm_val_std = np.std(Healthy_CSm_val, axis=0)/ stde_norm 

# No ET
NoET_CSp_val_mean = np.mean(NoET_CSp_val, axis=0) 
NoET_CSp_val_std = np.std(NoET_CSp_val, axis=0)/ stde_norm 
NoET_CSm_val_mean = np.mean(NoET_CSm_val, axis=0) 
NoET_CSm_val_std = np.std(NoET_CSm_val, axis=0)/ stde_norm 

# No IT
NoIT_CSp_val_mean = np.mean(NoIT_CSp_val, axis=0) 
NoIT_CSp_val_std = np.std(NoIT_CSp_val, axis=0)/ stde_norm 
NoIT_CSm_val_mean = np.mean(NoIT_CSm_val, axis=0) 
NoIT_CSm_val_std = np.std(NoIT_CSm_val, axis=0)/ stde_norm 

# No ET and IT
NoET_IT_CSp_val_mean = np.mean(NoET_IT_CSp_val, axis=0) 
NoET_IT_CSp_val_std = np.std(NoET_IT_CSp_val, axis=0)/ stde_norm 
NoET_IT_CSm_val_mean = np.mean(NoET_IT_CSm_val, axis=0) 
NoET_IT_CSm_val_std = np.std(NoET_IT_CSm_val, axis=0)/ stde_norm 

# Cortex independent expert performance for healthy 
Healthy_noCortex_acc_mean = np.mean(Healthy_noCortex_acc, axis=0) 
Healthy_noCortex_acc_std = np.std(Healthy_noCortex_acc, axis=0)/ stde_norm 
Healthy_cortexDep_acc_mean = np.mean(Healthy_cortexDep_acc, axis=0) 
Healthy_cortexDep_acc_std = np.std(Healthy_cortexDep_acc, axis=0)/ stde_norm 

striatal_acc_NoET_mean = np.mean(striatal_acc_NoET, axis=0) 
striatal_acc_NoET_std = np.std(striatal_acc_NoET, axis=0)/ stde_norm 

striatal_acc_NoIT_mean = np.mean(striatal_acc_NoIT, axis=0) 
striatal_acc_NoIT_std = np.std(striatal_acc_NoIT, axis=0) / stde_norm 

## =========== Plotting ================
CS_colors = ['tab:cyan','tab:olive']

#t = np.arange(0, len(Healthy_CSp_val[0]))
#conditions = ['start', 'early', 'mid', 'late', 'expert']
conditions = [0,1,2,3,4,5]
titles = ['Control', 'No IT', 'No ET'] 

CSp_val = [Healthy_CSp_val_mean, NoIT_CSp_val_mean, NoET_CSp_val_mean]
CSm_val = [Healthy_CSm_val_mean, NoIT_CSm_val_mean, NoET_CSm_val_mean]

CSp_val_std = [Healthy_CSp_val_std, NoIT_CSp_val_std, NoET_CSp_val_std]
CSm_val_std = [Healthy_CSm_val_std, NoIT_CSm_val_std, NoET_CSm_val_std]

# Experimental data
CSp_lick_rate = [Healthy_lick_CSp, NoIT_lick_CSp, NoET_lick_CSp]
CSm_lick_rate = [Healthy_lick_CSm, NoIT_lick_CSm, NoET_lick_CSm]

# Plot CS+ vs CS-  across conditions
for j in range(3):
    # Plot mice data
    axs[0,j].plot([1,2,3,4,5], CSp_lick_rate[j], color=CS_colors[0], alpha=0.8, label='CS+\n(mice)')
    axs[0,j].plot([1,2,3,4,5], CSm_lick_rate[j], color=CS_colors[1], alpha=0.8, label='CS-\n(mice)')
    # Plot model predictions
    axs[0, j].plot(conditions, CSp_val[j], linestyle='dashed', color=CS_colors[0], label='CS+\n(model)', marker='o', markersize=4,alpha=0.75)
    axs[0, j].fill_between(conditions, CSp_val[j] - CSp_val_std[j], CSp_val[j] + CSp_val_std[j], alpha=0.25, color=CS_colors[0])
    axs[0, j].plot(conditions, CSm_val[j], linestyle='dashed', color=CS_colors[1], label='CS-\n(model)', marker='o', markersize=4,alpha=0.75)
    axs[0, j].fill_between(conditions, CSm_val[j] - CSm_val_std[j], CSm_val[j] + CSm_val_std[j], alpha=0.25, color=CS_colors[1])
    #axs[0, j].errorbar(conditions, CSp_val[j], yerr=CSp_val_std[j], color=CS_colors[0], label='CS+\n(model)',capsize=3,fmt="--o",ecolor="black",markersize=4,alpha=0.5)
    #xs[0, j].errorbar(conditions, CSm_val[j], yerr=CSm_val_std[j], color=CS_colors[1], label='CS-\n(model)',capsize=3,fmt="--o",ecolor="black",markersize=4,alpha=0.5)

    axs[0, j].set_title(titles[j])
    if j!=0:
        axs[0, j].set_xlabel('Training sessions')
    if j==0:
        axs[0, j].set_ylabel('Association strength \n normalised lick rate (Hz)')
        #axs[0,j].set_xticks([])
    axs[0,j].spines['top'].set_visible(False)
    axs[0,j].spines['right'].set_visible(False)
    axs[0,j].set_ylim(0, 1)  # Example range from 0 to 20
    axs[0,j].set_xlim(0, 5)  
    #axs[0,j].set_yticks([0, 20, 40, 60, 80, 100])  # Example custom y-ticks
axs[0,0].legend(loc='lower left', bbox_to_anchor=(-0.1, -0.6), frameon=False,fontsize=font_s, ncol=4,  columnspacing=0.8)

# Plot mice data
axs[1,0].plot([1,2,3,4,5], NoET_IT_lick_CSp, color=CS_colors[0], alpha=0.8)
axs[1,0].plot([1,2,3,4,5], NoET_IT_lick_CSm, color=CS_colors[1], alpha=0.8)
# Plot model predictions
axs[1, 0].plot(conditions, NoET_IT_CSp_val_mean, linestyle='dashed', color=CS_colors[0], label='CS+\n(model)', marker='s', markersize=4,alpha=0.75)
axs[1, 0].fill_between(conditions, NoET_IT_CSp_val_mean - NoET_IT_CSp_val_std, NoET_IT_CSp_val_mean  + NoET_IT_CSp_val_std, alpha=0.25, color=CS_colors[0])
axs[1, 0].plot(conditions, NoET_IT_CSm_val_mean, linestyle='dashed', color=CS_colors[1], label='CS-\n(model)', marker='s', markersize=4,alpha=0.75)
axs[1, 0].fill_between(conditions, NoET_IT_CSm_val_mean - NoET_IT_CSm_val_std, NoET_IT_CSm_val_mean  + NoET_IT_CSm_val_std, alpha=0.25, color=CS_colors[1])
#axs[1, 0].errorbar(conditions, NoET_IT_CSp_val_mean, yerr=NoET_IT_CSp_val_std, color=CS_colors[0], label='CS+',capsize=3,fmt="--o",ecolor="black",markersize=4,alpha=0.5)
#axs[1, 0].errorbar(conditions, NoET_IT_CSm_val_mean, yerr=NoET_IT_CSm_val_std, color=CS_colors[1], label='CS+',capsize=3,fmt="--o",ecolor="black",markersize=4,alpha=0.5)

axs[1,0].set_title("No S1")
axs[1,0].set_xlabel('Training sessions')
axs[1,0].set_ylabel('Association strength \n normalised lick rate (Hz)')
axs[1,0].spines['top'].set_visible(False)
axs[1,0].spines['right'].set_visible(False)
axs[1,0].set_ylim(0, 1)  # Example range from 0 to 20
axs[1,0].set_xlim(0, 5)  

## ---------- Plot striatum accuracy without cortex --------
acc = [Healthy_noCortex_acc_mean, Healthy_cortexDep_acc_mean]
acc_std = [Healthy_noCortex_acc_std, Healthy_cortexDep_acc_std]
categoris = ['No S1', 'S1']
axs[1,1].bar(categoris, acc,  color='tab:gray', alpha=0.75)
axs[1,1].errorbar(categoris,acc, yerr=acc_std, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
axs[1,1].set_title('Expert performance')
axs[1,1].set_ylabel('Accuracy (%)')
axs[1,1].spines['top'].set_visible(False)
axs[1,1].spines['right'].set_visible(False)
axs[1,1].set_ylim(0, 100)  # Example range from 0 to 20
axs[1,1].set_yticks([0, 20, 40, 60, 80, 100])  # Example custom y-ticks

## ---------- Plot striatum accuracy without ET or IT ------
acc = [striatal_acc_NoIT_mean, striatal_acc_NoET_mean]
acc_std = [striatal_acc_NoIT_std, striatal_acc_NoET_std]
categoris = ['No IT', 'No ET']
axs[1,2].bar(categoris, acc, color=['tab:purple', 'tab:orange'], alpha=0.75)
axs[1,2].errorbar(categoris,acc, yerr=acc_std, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
axs[1,2].set_title('Learning accuracy without IT or ET')
axs[1,2].set_ylabel('Accuracy (%)')
axs[1,2].spines['top'].set_visible(False)
axs[1,2].spines['right'].set_visible(False)
axs[1,2].set_ylim(0, 100)  # Example range from 0 to 20
axs[1,2].set_yticks([0, 20, 40, 60, 80, 100])  # Example custom y-ticks


### =============== Run separate analysis where you plot IT feature overlap and performance for 3 other class comparisons =================
seeds = [62419, 87745, 55327, 31023, 21716]
s = seeds[0]
torch.manual_seed(s)
np.random.seed(s)
dev = 'cpu'
batch_s = 150 #500
dataset_name = "synthetic_data" #"synthetic_data" #mnist" #"cifar10"
pca = PCA(n_components=2)

titles = ['Stimulus A vs B', 'Stimulus A vs C', 'Stimulus A vs D']

## ----------- Plot feature overlap for class 0 vs 1 (high overlap) -------------
class_comp = 1
specific_classes = all_classes_comp[class_comp]
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s, specific_classes=specific_classes)

# Extra n. of channels, width and n. labels for the specific dataset
data_batch, label_batch = next(iter(training_data))

# make sure labels are {0,1}
max_label = torch.max(label_batch).item() 

label_batch = torch.floor(label_batch/max_label) # generalise rwd function to any two class labels

## ---- Initalise and load model (this has only to be done once) --------
input_s = data_batch.size()[1]
IT = IT_NN(input_s=input_s).to(dev)
IT_reps_s = IT.h_size
# Load model
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'..','..','models')
IT_model_dir = os.path.join(file_dir,f'{dataset_name}_IT_model.pt')
IT.load_state_dict(torch.load(IT_model_dir,  map_location=dev))
## ----------------------------------------


## Get IT features for specific class comparison
_ , IT_features = IT(data_batch)
IT_features = IT_features.detach().numpy() 
mean_IT_features = np.mean(IT_features, axis=0)
norm_IT_features = IT_features - mean_IT_features
## Run PCA
components = pca.fit(norm_IT_features) # Fit the model with X and apply the dimensionality reduction on X
print(np.sum(pca.explained_variance_ratio_))

# Projection data based on API
X_hat = pca.transform(norm_IT_features) # X is projected on the first principal components previously extracted from a training set

# ---- Manual projection of data (fun exercise) -----
#components = pca.components_.T # transpose to have basis in standard form
#comp_coord = np.linalg.inv(components.T @ components) @ components.T @ norm_IT_features.T
#print(np.allclose(X_hat, comp_coord.T))
# ---------------------------

# extract features for CS+ vs CS-
CSp_features_pca = X_hat[label_batch==1]
CSm_features_pca = X_hat[label_batch==0]

## Plot PCA for class 0 vs class 1
axs[2,0].scatter(CSp_features_pca[:,0], CSp_features_pca[:,1],label='CS+', color=CS_colors[0], s=10)
axs[2,0].scatter(CSm_features_pca[:,0], CSm_features_pca[:,1],label='CS-', color=CS_colors[1], s=10)
axs[2,0].set_title(f"IT feature overlap \n {titles[0]}")
axs[2,0].set_xlabel('1st component')
axs[2,0].set_ylabel('2nd component')
axs[2,0].spines['top'].set_visible(False)
axs[2,0].spines['right'].set_visible(False)
axs[2,0].set_xticklabels([])
axs[2,0].set_yticklabels([])

## ----------- Plot feature overlap for class 0 vs 5 (mid overlap) -------------
class_comp = 3
specific_classes = all_classes_comp[class_comp]
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s, specific_classes=specific_classes)

# Extra n. of channels, width and n. labels for the specific dataset
data_batch, label_batch = next(iter(training_data))

# make sure labels are {0,1}
max_label = torch.max(label_batch).item() 

label_batch = torch.floor(label_batch/max_label) # generalise rwd function to any two class labels



## ----- extract IT feature overlap
## Get IT features
_ , IT_features = IT(data_batch)

IT_features = IT_features.detach().numpy() 

mean_IT_features = np.mean(IT_features, axis=0)

norm_IT_features = IT_features - mean_IT_features

## ----- Run PCA -----
pca = PCA(n_components=2)
components = pca.fit(norm_IT_features) # Fit the model with X and apply the dimensionality reduction on X
print(np.sum(pca.explained_variance_ratio_))

# Projection data based on API
X_hat = pca.transform(norm_IT_features) # X is projected on the first principal components previously extracted from a training set

# extract features for CS+ vs CS-
CSp_features_pca = X_hat[label_batch==1]
CSm_features_pca = X_hat[label_batch==0]

## ------ Plot PCA 2 components -------
axs[2,1].scatter(CSp_features_pca[:,0], CSp_features_pca[:,1],label='CS+', color=CS_colors[0], s=10)
axs[2,1].scatter(CSm_features_pca[:,0], CSm_features_pca[:,1],label='CS-', color=CS_colors[1], s=10)
axs[2,1].set_title(f"IT feature overlap \n {titles[1]}")
axs[2,1].set_xlabel('1st component')
axs[2,1].set_ylabel('2nd component')
axs[2,1].spines['top'].set_visible(False)
axs[2,1].spines['right'].set_visible(False)
axs[2,1].set_xticklabels([])
axs[2,1].set_yticklabels([])

## ----------- Plot feature overlap for class 0 vs 4 (No overlap) -------------
class_comp = 2
specific_classes = all_classes_comp[class_comp]
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s, specific_classes=specific_classes)

# Extra n. of channels, width and n. labels for the specific dataset
data_batch, label_batch = next(iter(training_data))

# make sure labels are {0,1}
max_label = torch.max(label_batch).item() 

label_batch = torch.floor(label_batch/max_label) # generalise rwd function to any two class labels

## Get IT features
_ , IT_features = IT(data_batch)
IT_features = IT_features.detach().numpy() 
mean_IT_features = np.mean(IT_features, axis=0)
norm_IT_features = IT_features - mean_IT_features
## Run PCA
pca = PCA(n_components=2)
components = pca.fit(norm_IT_features) # Fit the model with X and apply the dimensionality reduction on X
print(np.sum(pca.explained_variance_ratio_))

# Projection data based on API
X_hat = pca.transform(norm_IT_features) # X is projected on the first principal components previously extracted from a training set

# extract features for CS+ vs CS-
CSp_features_pca = X_hat[label_batch==1]
CSm_features_pca = X_hat[label_batch==0]

## Plot PCA for class 0 vs class 1
axs[2,2].scatter(CSp_features_pca[:,0], CSp_features_pca[:,1],label='CS+', color=CS_colors[0], s=10)
axs[2,2].scatter(CSm_features_pca[:,0], CSm_features_pca[:,1],label='CS-', color=CS_colors[1], s=10)
axs[2,2].set_title(f"IT feature overlap \n {titles[2]}")
axs[2,2].set_xlabel('1st component')
axs[2,2].set_ylabel('2nd component')
axs[2,2].spines['top'].set_visible(False)
axs[2,2].spines['right'].set_visible(False)
axs[2,2].set_xticklabels([])
axs[2,2].set_yticklabels([])

## -------- Plot striatal loss for each class comparison ---------
class_plot_order = [1, 3, 2] # plot in same order as above (i.e., from more to less overlap)
class_CSp_mean = []
class_CSm_mean = []
class_CSp_std = []
class_CSm_std = []
for c in class_plot_order:
    seed_CSp_val = []
    seed_CSm_val = []
    specific_classes = all_classes_comp[c]
    for s in seeds:
        data_1 = f'data/VB_{dataset_name}_LickData_seed_{s}_class_{specific_classes[0]}_vs_{specific_classes[1]}.pt'
        data_1 = torch.load(data_1)

        ## Extract values
        seed_CSp_val.append(data_1['CSp_rwd'])
        seed_CSm_val.append(data_1['CSm_rwd'])
    
    class_CSp_mean.append(np.mean(seed_CSp_val,axis=0))
    class_CSm_mean.append(np.mean(seed_CSm_val,axis=0))
    class_CSp_std.append(np.std(seed_CSp_val,axis=0)/ stde_norm)
    class_CSm_std.append(np.std(seed_CSm_val,axis=0)/ stde_norm)

# Extract session indexes
class_CSp_mean = np.array(class_CSp_mean)[:,indx_learning_stages]
class_CSm_mean = np.array(class_CSm_mean)[:,indx_learning_stages]
class_CSp_std = np.array(class_CSp_std)[:,indx_learning_stages]
class_CSm_std = np.array(class_CSm_std)[:,indx_learning_stages]

## Insert baseline at trial zero since saved results from averaged of first 100 trials
class_CSp_mean[:,0] = 0 
class_CSm_mean[:,0] = 0 
class_CSp_std[:,0] = 0 
class_CSm_std[:,0] = 0 

# Plot CS+ vs CS-  across stimuli
for j in range(3):
    axs[3, j].plot(conditions, class_CSp_mean[j], linestyle='dashed', color=CS_colors[0], label='CS+\n(model)',marker='o', markersize=4, alpha=0.75)
    axs[3, j].fill_between(conditions, class_CSp_mean[j] - class_CSp_std[j], class_CSp_mean[j] + class_CSp_std[j], alpha=0.25, color=CS_colors[0])
    axs[3, j].plot(conditions, class_CSm_mean[j], linestyle='dashed', color=CS_colors[1], label='CS+\n(model)',marker='o', markersize=4, alpha=0.75)
    axs[3, j].fill_between(conditions, class_CSm_mean[j] - class_CSm_std[j], class_CSm_mean[j] + class_CSm_std[j], alpha=0.25, color=CS_colors[1])
    #axs[3, j].errorbar(conditions, class_CSp_mean[j], yerr=class_CSp_std[j], color=CS_colors[0], label='CS+\n(model)',capsize=3,fmt="r--o",ecolor="black",markersize=4,alpha=0.5)
    #axs[3, j].errorbar(conditions, class_CSm_mean[j], yerr=class_CSm_std[j], color=CS_colors[1], label='CS-\n(model)',capsize=3,fmt="r--o",ecolor="black",markersize=4,alpha=0.5)
    axs[3, j].set_title('Peformance \n'+titles[j])
    axs[3, j].set_xlabel('Training sessions')
    if j==0:
        axs[3, j].set_ylabel('Association strength')
        #axs[0,j].set_xticks([])
    axs[3,j].spines['top'].set_visible(False)
    axs[3,j].spines['right'].set_visible(False)
    axs[3,j].set_ylim(0, 1)  # Example range from 0 to 20
    axs[3,j].set_xlim(0, 5)  


# 3D plot
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(CSp_features_tsne[:,0], CSp_features_tsne[:,1], CSp_features_tsne[:,2])
#ax.scatter(CSm_features_tsne[:,0], CSm_features_tsne[:,1], CSm_features_tsne[:,2])


#axs[2,0].remove()

#fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.75),#subplot_kw=dict(projection='3d'),
#                        gridspec_kw={'wspace': 0.65, 'hspace': 0.6, 'left': 0.075, 'right': 0.95, 'bottom': 0.1,'top': 0.94})

#ax.scatter(CSp_features_pca[:,0], CSp_features_pca[:,1], CSp_features_pca[:,2],label='CS+')
#ax.scatter(CSm_features_pca[:,0], CSm_features_pca[:,1], CSm_features_pca[:,2],label='CS-')
#ax.set_zticklabels([])
#ax.set_yticklabels([])
#ax.set_xticklabels([])
#ax.legend(loc='lower left', bbox_to_anchor=(0.8, 0))
# Turn off tick labels

plt.show()


# Adjust the layout to prevent overlapping
#plt.tight_layout()

# Display the plot
#plt.savefig('/Users/px19783/Desktop/VB_ET_IT_draft.eps', format='eps', dpi=1400)
#plt.savefig('/Users/px19783/Desktop/VB_ET_IT_draft.png', format='png', dpi=1400)

 

