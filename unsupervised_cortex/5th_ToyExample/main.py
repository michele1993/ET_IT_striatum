import torch
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import get_data
from utils import setup_logger
import logging
from TrainingLoop import TrainingLoop
 
""" 
In this Toy model, I model IT as CNN features of an encoder trained with unsuperivsed learning (by including a bottleneck and decoder for training IT features only), 
, while I assume ET to build a rwd prediction from the IT representations. 
The ET predictions are then passed to the striatum as target for training (instead of actual rwds) as well as input to the striatum (if ET_as_input=True).
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
ET_feedback = True
IT_feedback = True
cortex_ET_s = 256 # model cortex as a large (powerful) NN
striatal_h_state = 20 # model striatum as a small (linear) NN
impairCortex_afterLearning = True # At the moment assessed on test data
specific_classes = [0,1] # only ask two discriminate between two classes

# Training variables
epocs = 1
batch_s = 64
striatum_training_delay = 0 # delay training of the striatum by n. epocs, to allow cortex to learn good reprs. first
ET_ln_rate = 1e-3
striatal_ln_rate = 1e-4 #1e-3 #1e-5


# Get data organised in batches 
assert specific_classes is not None, "The reward function only works for two specific classess"
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s, specific_classes=specific_classes)

# Allows to use correct rwd function depending on
max_label = np.max(specific_classes)

# Initialise training loop
trainingloop = TrainingLoop(dataset_name=dataset_name, training_data=training_data, test_data=test_data, n_labels=n_labels, 
                            max_label=max_label, striatum_training_delay=striatum_training_delay, 
                            ET_ln_rate=ET_ln_rate, cortex_ET_s = cortex_ET_s, 
                            striatal_ln_rate=striatal_ln_rate, striatal_h_state=striatal_h_state, 
                            IT_feedback=IT_feedback, ET_feedback=ET_feedback, device=dev)

tot_CSp_action_p = []
tot_CSm_action_p = []
tot_mean_rwd = []
for e in range(epocs):
    CSp_action_p, CSm_action_p, mean_rwd = trainingloop.train(e)
    tot_CSp_action_p.append(CSp_action_p)
    tot_CSm_action_p.append(CSm_action_p)
    tot_mean_rwd.append(mean_rwd)

tot_CSp_action_p = np.array(tot_CSm_action_p).reshape(-1)
tot_CSm_action_p = np.array(tot_CSp_action_p).reshape(-1)
tot_mean_rwd = np.array(tot_mean_rwd).reshape(-1)

#print(tot_CSp_action_p, tot_CSm_action_p, tot_mean_rwd)

## ----- Useful plots ------
#trainingloop.plot_imgs()
#trainingloop.plot_latent_space_with_labels(num_classes=n_labels, specific_classes=specific_classes)
## -------------------------
## ------ Test performance ---------
striatal_rwd_performance,  cortex_rwd_performance, actions = trainingloop.test_performance(impairCortex_afterLearning)
striatal_test_rwd_acc = np.round(sum(striatal_rwd_performance)/len(striatal_rwd_performance),decimals=2)
cortex_test_rwd_acc = np.round(sum(cortex_rwd_performance)/len(cortex_rwd_performance),decimals=2)
mean_test_action = sum(actions)/len(actions)
logging.info(f"*** | Striatal test rwd accuracy:  {striatal_test_rwd_acc*100}% | Cortex test rwd accuracy:  {cortex_test_rwd_acc*100}% | Test mean action: {mean_test_action}***")
## ---------------------

## ------- Save files -------
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results')

data = f'{dataset_name}_LickData'
if not ET_feedback ==0:
    data += '_noET_'
if not IT_feedback ==1:    
    data += '_noIT_'
data += '.pt'

model_dir = os.path.join(file_dir,data)

if save_file:
    # Create directory if it did't exist before
    os.makedirs(file_dir, exist_ok=True)
    torch.save({
        "Cortex_acc": cortex_test_rwd_acc,
        "Striatal_acc": striatal_test_rwd_acc,
        "CSp_action_p": tot_CSp_action_p,
        "CSm_action_p": tot_CSm_action_p,
    }, model_dir)
