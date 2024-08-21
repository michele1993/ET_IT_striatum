import torch
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import get_data
from utils import setup_logger
import logging
from TrainingLoop import TrainingLoop
import matplotlib.pyplot as plt
 
""" 
In this Toy model, I model IT as CNN features of an encoder trained with unsuperivsed learning (by including a bottleneck and decoder for training IT features only), 
, while I assume ET to build a rwd prediction from the IT representations. 
Crucially, the IT (CNN) network is pre-trained based on all classes and then loaded for the reward association task as it is, without undergoing any further training.
The ET reward predictions are used as a baseline in the REINFORCE update to update the policy based on a reward prediction error
"""

## ---- Set seed for reproducibility purposes
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
save_file = True

# Select correct device
if torch.cuda.is_available():
    dev='cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): ## for MAC GPU usage
    dev='mps'
else:
    dev='cpu'

setup_logger()

## Experiment variables
dataset_name = "cifar10" #"synthetic_data" #mnist" #"cifar10"
ET_feedback = False
IT_feedback = False
cortex_ET_s = 56 # model cortex as a large (powerful) NN
striatal_h_state = 116 # model striatum as a small (linear) NN
specific_classes = [0,1] # only ask two discriminate between two classes

# Training variables
epocs = 25
batch_s = 64
ET_ln_rate = 0.0001
striatal_ln_rate = 1e-5 #1e-3 #1e-5


# Get data organised in batches 
assert specific_classes is not None, "The reward function only works for two specific classess"
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s, specific_classes=specific_classes)

# Allows to use correct rwd function depending on
max_label = np.max(specific_classes)

# Initialise training loop
trainingloop = TrainingLoop(dataset_name=dataset_name, training_data=training_data, test_data=test_data, n_labels=n_labels, 
                            max_label=max_label, ET_ln_rate=ET_ln_rate, cortex_ET_s = cortex_ET_s, 
                            striatal_ln_rate=striatal_ln_rate, striatal_h_state=striatal_h_state, 
                            IT_feedback=IT_feedback, ET_feedback=ET_feedback, device=dev)

tot_rwd_acc = []
tot_noCortex_rwd_acc = []
tot_CS_p_rwd = []
tot_CS_m_rwd = []
for e in range(epocs):
    tot_striatal_rwd_loss, tot_striatal_noCortex_loss, CS_p_rwd, CS_m_rwd = trainingloop.train(e)
    tot_rwd_acc.append(tot_striatal_rwd_loss)
    tot_noCortex_rwd_acc.append(tot_striatal_noCortex_loss)
    tot_CS_p_rwd.append(CS_p_rwd)
    tot_CS_m_rwd.append(CS_m_rwd)

tot_rwd_acc = np.array(tot_rwd_acc).reshape(-1)
tot_noCortex_rwd_acc = np.array(tot_noCortex_rwd_acc).reshape(-1)
tot_CS_p_rwd = np.array(tot_CS_p_rwd).reshape(-1)
tot_CS_m_rwd = np.array(tot_CS_m_rwd).reshape(-1)

## ----- Useful plots ------
#trainingloop.plot_imgs()
#trainingloop.plot_latent_space_with_labels(num_classes=n_labels, specific_classes=specific_classes)
## -------------------------
## ------ Test performance ---------
cortexDep_rwd_performance, noCortex_rwd_performance = trainingloop.test_performance()
cortexDep_test_rwd_acc = np.round(sum(cortexDep_rwd_performance)/len(cortexDep_rwd_performance),decimals=2)
noCortex_test_rwd_acc = np.round(sum(noCortex_rwd_performance)/len(noCortex_rwd_performance),decimals=2)
logging.info(f"*** | Cortex-dependent test rwd accuracy:  {cortexDep_test_rwd_acc*100}% | No cortex test rwd accuracy:  {noCortex_test_rwd_acc*100}% | ***")
## ---------------------

## ------ Plot training curve -------
t = np.arange(1,len(tot_CS_p_rwd)+1)

plt.plot(t, tot_CS_p_rwd)
plt.plot(t, tot_CS_m_rwd)
plt.show()


## ------- Save files -------
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results')

data = f'VB_{dataset_name}_LickData'
if not ET_feedback:
    data += '_noET_'
if not IT_feedback:    
    data += '_noIT_'
data += '.pt'

model_dir = os.path.join(file_dir,data)

if save_file:
    # Create directory if it did't exist before
    os.makedirs(file_dir, exist_ok=True)
    torch.save({
        "Striatum_param": trainingloop.striatum.state_dict(), 
        "NoCortex_acc": noCortex_test_rwd_acc,
        "Striatal_acc": cortexDep_test_rwd_acc,
        "CSp_rwd": tot_CS_p_rwd,
        "CSm_rwd": tot_CS_m_rwd,
    }, model_dir)
