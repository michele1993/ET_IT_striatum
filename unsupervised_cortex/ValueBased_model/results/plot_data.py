import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils import get_data
from unsupervised_cortex.preTrain_FF_cortex.IT_NN import IT_NN
from unsupervised_cortex.ValueBased_model.Striatum_lNN import Striatum_lNN



seeds = [62419, 87745, 55327, 31023, 21716]

font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s

# Create a figure and a set of subplots
fig, axs = plt.subplots(3, 3, figsize=(7, 5),
                        gridspec_kw={'wspace': 0.65, 'hspace': 0.6, 'left': 0.075, 'right': 0.95, 'bottom': 0.1,'top': 0.94})



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
normaliser = np.max(NoET_lick_CSp)

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

for s in seeds:
    data_1 = f'data/VB_{dataset_name}_LickData_seed_{s}'
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
indx_learning_stages = [0, 9, 19, 29, 39, 49]
Healthy_CSp_val = np.array(Healthy_CSp_val)[:,indx_learning_stages] 
Healthy_CSm_val = np.array(Healthy_CSm_val)[:,indx_learning_stages]
NoET_CSp_val = np.array(NoET_CSp_val)[:,indx_learning_stages]
NoET_CSm_val = np.array(NoET_CSm_val)[:,indx_learning_stages]
NoIT_CSp_val = np.array(NoIT_CSp_val)[:,indx_learning_stages]
NoIT_CSm_val = np.array(NoIT_CSm_val)[:,indx_learning_stages]
NoET_IT_CSp_val = np.array(NoET_IT_CSp_val)[:,indx_learning_stages]
NoET_IT_CSm_val = np.array(NoET_IT_CSm_val)[:,indx_learning_stages]

Healthy_noCortex_acc = np.array(Healthy_noCortex_acc)
Healthy_cortexDep_acc = np.array(Healthy_cortexDep_acc)
striatal_acc_NoET = np.array(striatal_acc_NoET)
striatal_acc_NoIT = np.array(striatal_acc_NoIT)

## Compute mean and std
# Healthy
Healthy_CSp_val_mean = np.mean(Healthy_CSp_val, axis=0) 
Healthy_CSp_val_std = np.std(Healthy_CSp_val, axis=0) 
Healthy_CSm_val_mean = np.mean(Healthy_CSm_val, axis=0) 
Healthy_CSm_val_std = np.std(Healthy_CSm_val, axis=0) 

# No ET
NoET_CSp_val_mean = np.mean(NoET_CSp_val, axis=0) 
NoET_CSp_val_std = np.std(NoET_CSp_val, axis=0) 
NoET_CSm_val_mean = np.mean(NoET_CSm_val, axis=0) 
NoET_CSm_val_std = np.std(NoET_CSm_val, axis=0) 

# No IT
NoIT_CSp_val_mean = np.mean(NoIT_CSp_val, axis=0) 
NoIT_CSp_val_std = np.std(NoIT_CSp_val, axis=0) 
NoIT_CSm_val_mean = np.mean(NoIT_CSm_val, axis=0) 
NoIT_CSm_val_std = np.std(NoIT_CSm_val, axis=0) 

# No ET and IT
NoET_IT_CSp_val_mean = np.mean(NoET_IT_CSp_val, axis=0) 
NoET_IT_CSp_val_std = np.std(NoET_IT_CSp_val, axis=0) 
NoET_IT_CSm_val_mean = np.mean(NoET_IT_CSm_val, axis=0) 
NoET_IT_CSm_val_std = np.std(NoET_IT_CSm_val, axis=0) 

# Cortex independent expert performance for healthy 
Healthy_noCortex_acc_mean = np.mean(Healthy_noCortex_acc, axis=0) 
Healthy_noCortex_acc_std = np.std(Healthy_noCortex_acc, axis=0) 
Healthy_cortexDep_acc_mean = np.mean(Healthy_cortexDep_acc, axis=0) 
Healthy_cortexDep_acc_std = np.std(Healthy_cortexDep_acc, axis=0) 

striatal_acc_NoET_mean = np.mean(striatal_acc_NoET, axis=0) 
striatal_acc_NoET_std = np.std(striatal_acc_NoET, axis=0) 

striatal_acc_NoIT_mean = np.mean(striatal_acc_NoIT, axis=0) 
striatal_acc_NoIT_std = np.std(striatal_acc_NoIT, axis=0) 

## =========== Plotting ================
CS_colors = ['tab:cyan','tab:olive']

#t = np.arange(0, len(Healthy_CSp_val[0]))
#conditions = ['start', 'early', 'mid', 'late', 'expert']
conditions = [0,1,2,3,4,5]
titles = ['Healthy', 'No IT', 'No ET'] 

CSp_val = [Healthy_CSp_val_mean, NoIT_CSp_val_mean, NoET_CSp_val_mean]
CSm_val = [Healthy_CSm_val_mean, NoIT_CSm_val_mean, NoET_CSm_val_mean]

CSp_val_std = [Healthy_CSp_val_std, NoIT_CSp_val_std, NoET_CSp_val_std]
CSm_val_std = [Healthy_CSm_val_std, NoIT_CSm_val_std, NoET_CSm_val_std]

# Experimental data
CSp_lick_rate = [Healthy_lick_CSp, NoIT_lick_CSp, NoET_lick_CSp]
CSm_lick_rate = [Healthy_lick_CSm, NoIT_lick_CSm, NoET_lick_CSm]

# Plot something on each subplot
for j in range(3):
    axs[0, j].errorbar(conditions, CSp_val[j], yerr=CSp_val_std[j], color=CS_colors[0], label='CS+',capsize=3,fmt="r--o",ecolor="black",markersize=4,alpha=0.75)
    axs[0, j].errorbar(conditions, CSm_val[j], yerr=CSm_val_std[j], color=CS_colors[1], label='CS-',capsize=3,fmt="r--o",ecolor="black",markersize=4,alpha=0.75)
    axs[0,j].plot([1,2,3,4,5], CSp_lick_rate[j])
    axs[0,j].plot([1,2,3,4,5], CSm_lick_rate[j])
    axs[0, j].set_title(titles[j])
    axs[0, j].set_xlabel('Days')
    axs[0, j].set_ylabel('Predicted value')
    axs[0,j].spines['top'].set_visible(False)
    axs[0,j].spines['right'].set_visible(False)
    axs[0,j].set_ylim(0, 1)  # Example range from 0 to 20
    axs[0,j].set_xlim(0, 5)  
    #axs[0,j].set_yticks([0, 20, 40, 60, 80, 100])  # Example custom y-ticks
axs[0,0].legend(loc='lower left', bbox_to_anchor=(1.05, -0.4), frameon=False,fontsize=font_s)

axs[1, 0].errorbar(conditions, NoET_IT_CSp_val_mean, yerr=NoET_IT_CSp_val_std, color=CS_colors[0], label='CS+',capsize=3,fmt="r--o",ecolor="black",markersize=4,alpha=0.75)
axs[1, 0].errorbar(conditions, NoET_IT_CSm_val_mean, yerr=NoET_IT_CSm_val_std, color=CS_colors[1], label='CS+',capsize=3,fmt="r--o",ecolor="black",markersize=4,alpha=0.75)
axs[1,0].plot([1,2,3,4,5], NoET_IT_lick_CSp)
axs[1,0].plot([1,2,3,4,5], NoET_IT_lick_CSm)
axs[1,0].set_title("No S1")
axs[1,0].set_xlabel('Days')
axs[1,0].set_ylabel('Predicted value')
axs[1,0].spines['top'].set_visible(False)
axs[1,0].spines['right'].set_visible(False)
axs[1,0].set_ylim(0, 1)  # Example range from 0 to 20
axs[1,0].set_xlim(0, 5)  

## ---------- Plot striatum accuracy without cortex --------
acc = [Healthy_noCortex_acc_mean*100, Healthy_cortexDep_acc_mean*100]
categoris = ['No Cortex', 'Cortex']
axs[1,1].bar(categoris, acc, color='tab:gray', alpha=0.75)
axs[1,1].set_title('Expert performance')
axs[1,1].set_ylabel('Accuracy (%)')
axs[1,1].spines['top'].set_visible(False)
axs[1,1].spines['right'].set_visible(False)
axs[1,1].set_ylim(0, 100)  # Example range from 0 to 20
axs[1,1].set_yticks([0, 20, 40, 60, 80, 100])  # Example custom y-ticks

## ---------- Plot striatum accuracy without ET or IT ------
acc = [striatal_acc_NoIT_mean*100, striatal_acc_NoET_mean*100]
categoris = ['No IT', 'No ET']
axs[1,2].bar(categoris, acc, color=['tab:purple', 'tab:orange'], alpha=0.75)
axs[1,2].set_title('Final learning accuracy without IT or ET')
axs[1,2].set_ylabel('Accuracy (%)')
axs[1,2].spines['top'].set_visible(False)
axs[1,2].spines['right'].set_visible(False)
axs[1,2].set_ylim(0, 100)  # Example range from 0 to 20
axs[1,2].set_yticks([0, 20, 40, 60, 80, 100])  # Example custom y-ticks

## ----------- Plot feature overlap -------------
seeds = [62419, 87745, 55327, 31023, 21716]
s = seeds[0]
torch.manual_seed(s)
np.random.seed(s)
dev = 'cpu'
batch_s = 500
specific_classes = [0,5]#[0,2] # only ask two discriminate between two classes
dataset_name = "synthetic_data" #"synthetic_data" #mnist" #"cifar10"
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s, specific_classes=specific_classes)


# Extra n. of channels, width and n. labels for the specific dataset
data_batch, label_batch = next(iter(training_data))

# make sure labels are {0,1}
max_label = torch.max(label_batch).item() 

label_batch = torch.floor(label_batch/max_label) # generalise rwd function to any two class labels

# Initialise and load pretrained IT model
dataset_name == 'synthetic_data'
input_s = data_batch.size()[1]
IT = IT_NN(input_s=input_s).to(dev)
IT_reps_s = IT.h_size


# Load model
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'..','..','models')
IT_model_dir = os.path.join(file_dir,f'{dataset_name}_IT_model.pt')
IT.load_state_dict(torch.load(IT_model_dir,  map_location=dev))

# Initialise Striatum
striatum = Striatum_lNN(input_s=input_s, IT_inpt_s=IT_reps_s, ln_rate=0, h_size=116).to(dev)


## ----- extract IT feature overlap
## Get IT features
_ , IT_features = IT(data_batch)

IT_features = IT_features.detach().numpy() 

mean_IT_features = np.mean(IT_features, axis=0)

norm_IT_features = IT_features - mean_IT_features




## ----- Run PCA -----
pca = PCA(n_components=3)
components = pca.fit(norm_IT_features) # Fit the model with X and apply the dimensionality reduction on X
print(pca.explained_variance_ratio_)
#exit()

# Projection data based on API
X_hat = pca.transform(norm_IT_features) # X is projected on the first principal components previously extracted from a training set

# ---- Manual projection of data (fun exercise) -----
#components = pca.components_.T # transpose to have basis in standard form
#comp_coord = np.linalg.inv(components.T @ components) @ components.T @ norm_IT_features.T
#print(np.allclose(X_hat, comp_coord.T))
# ---------------------------

CSp_features_hat = X_hat[label_batch==1]
CSm_features_hat = X_hat[label_batch==0]

## ------ Run TSNE -------

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(CSp_features_hat[:,0], CSp_features_hat[:,1], CSp_features_hat[:,2])
ax.scatter(CSm_features_hat[:,0], CSm_features_hat[:,1], CSm_features_hat[:,2])
plt.show()

exit()

fig = px.scatter_matrix(
    components,
    labels=label_batch,
    dimensions=range(1),
)




# Adjust the layout to prevent overlapping
#plt.tight_layout()

# Display the plot
plt.show()

#plt.savefig('/Users/px19783/Desktop/VB_ET_IT_draft', format='png', dpi=1400)

 

