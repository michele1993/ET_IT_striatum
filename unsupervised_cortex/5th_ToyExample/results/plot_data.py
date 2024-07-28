import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 3, figsize=(7, 3.5),
                        gridspec_kw={'wspace': 0.65, 'hspace': 0.6, 'left': 0.075, 'right': 0.95, 'bottom': 0.1,'top': 0.94})

dataset_name = 'cifar10'

file_dir = os.path.dirname(os.path.abspath(__file__))
data_1 = f'{dataset_name}_LickData'
data_2 = data_1 + '_noET_.pt'
data_3 = data_1 + '_noIT_.pt'
data_1 += '.pt'

data_1 = torch.load(data_1)
data_2 = torch.load(data_2)
data_3 = torch.load(data_3)

## ----------- Plot action probs -------------
## Extract action prob
Healthy_CSp_act_p = data_1['CSp_action_p']
Healthy_CSm_act_p = data_1['CSm_action_p']

NoET_CSp_act_p = data_2['CSp_action_p']
NoET_CSm_act_p = data_2['CSm_action_p']

NoIT_CSp_act_p = data_3['CSp_action_p']
NoIT_CSm_act_p = data_3['CSm_action_p']

CSp_action_p = [Healthy_CSp_act_p, NoIT_CSp_act_p, NoET_CSp_act_p]
CSm_action_p = [Healthy_CSm_act_p, NoIT_CSm_act_p, NoET_CSm_act_p]


t = np.arange(1, len(Healthy_CSp_act_p)+1)
titles = ['Healthy', 'No IT', 'No ET'] 

# Plot something on each subplot
for j in range(3):
    axs[0, j].plot(t, CSp_action_p[j], color='tab:blue', label='CS+', alpha=1)
    axs[0, j].plot(t, CSm_action_p[j], color='tab:orange', label='CS-', alpha=1)
    axs[0, j].set_title(titles[j])
    axs[0, j].set_xlabel('Trials x 25')
    axs[0, j].set_ylabel('Lick probability')
    axs[0,j].spines['top'].set_visible(False)
    axs[0,j].spines['right'].set_visible(False)
    axs[0,j].set_ylim(0, 1)  # Example range from 0 to 20
    #axs[0,j].set_yticks([0, 20, 40, 60, 80, 100])  # Example custom y-ticks
axs[0,0].legend(loc='lower left', bbox_to_anchor=(1.05, -0.4), frameon=False,fontsize=font_s)
## ---------- Plot rwd estimate --------
expect_rwd = data_1['Expect_rwd']

t = np.arange(1, len(expect_rwd)+1)

axs[1, 0].plot(t, expect_rwd, color='tab:green')
axs[1,0].set_title('Expected rwd (ET)')
axs[1,0].set_xlabel('Trials x 25')
axs[1,0].set_ylabel('Magnitude')
axs[1,0].spines['top'].set_visible(False)
axs[1,0].spines['right'].set_visible(False)

## ---------- Plot striatum accuracy without cortex --------
cortex_acc = data_1['Cortex_acc']
striatal_acc = data_1['Striatal_acc']


acc = [cortex_acc*100, striatal_acc*100]
categoris = ['Cortex', 'Striatum']
axs[1,1].bar(categoris, acc)
axs[1,1].set_title('Striatum without cortex')
axs[1,1].set_ylabel('Accuracy (%)')
axs[1,1].spines['top'].set_visible(False)
axs[1,1].spines['right'].set_visible(False)
axs[1,1].set_ylim(0, 100)  # Example range from 0 to 20
axs[1,1].set_yticks([0, 20, 40, 60, 80, 100])  # Example custom y-ticks

## ---------- Plot striatum accuracy without ET or IT ------
striatal_acc_NoET = data_2['Striatal_acc']
striatal_acc_NoIT = data_3['Striatal_acc']


acc = [striatal_acc_NoIT*100, striatal_acc_NoET*100]
categoris = ['No IT', 'No ET']
axs[1,2].bar(categoris, acc)
axs[1,2].set_title('Accuracy without IT or ET')
axs[1,2].set_ylabel('Accuracy (%)')
axs[1,2].spines['top'].set_visible(False)
axs[1,2].spines['right'].set_visible(False)
axs[1,2].set_ylim(0, 100)  # Example range from 0 to 20
axs[1,2].set_yticks([0, 20, 40, 60, 80, 100])  # Example custom y-ticks

# Adjust the layout to prevent overlapping
#plt.tight_layout()

# Display the plot
plt.show()

#plt.savefig('/Users/px19783/Desktop/ET_IT_draft', format='png', dpi=1400)

 

