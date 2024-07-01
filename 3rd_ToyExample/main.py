import torch
import numpy as np
from TrainingLoop import TrainingLoop
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data
from utils import setup_logger
import logging
 
""" 
In this Toy model, I model IT as CNN features of an encoder trained with unsuperivsed learning (by including a bottleneck and decoder for training IT features only), 
, while I assume ET to build a rwd prediction from the IT representations. 
The ET predictions are then passed to the striatum as target for training (instead of actual rwds) as well as input to the striatum (if ET_as_input=True).
"""

## ---- Set seed for reproducibility purposes
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

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
ET_as_input = True # controls whether ET rwd predictions are also passed to the striatum as inputs
IT_feedback = True
cortex_bottleneck_s = 56 # model cortex as a large (powerful) NN
cortex_ET_s = 25 # model cortex as a large (powerful) NN
striatal_h_state = 25 # model striatum as a small (linear) NN
impairCortex_afterLearning = False # At the moment assessed on test data
specific_classes = [0,1] # only ask two discriminate between two classes

# Training variables
epocs = 3#0#00
batch_s = 32
striatum_training_delay = 0 # delay training of the striatum by n. epocs, to allow cortex to learn good reprs. first
cortex_ln_rate = 5e-4
striatal_ln_rate = 5e-5


# Get data organised in batches 
assert specific_classes is not None, "The reward function only works for two specific classess"
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s, specific_classes=specific_classes)

# Allows to use correct rwd function depending on
max_label = np.max(specific_classes)

# Initialise training loop
trainingloop = TrainingLoop(training_data=training_data, test_data=test_data, n_labels=n_labels, max_label=max_label, striatum_training_delay=striatum_training_delay, 
                            cortex_ln_rate=cortex_ln_rate, cortex_bottleneck_s=cortex_bottleneck_s, cortex_ET_s = cortex_ET_s, 
                            striatal_ln_rate=striatal_ln_rate, striatal_h_state=striatal_h_state, 
                            IT_feedback=IT_feedback, ET_feedback=ET_as_input, device=dev)


for e in range(epocs):
    trainingloop.train(e)

## ----- Useful plots ------
#trainingloop.plot_imgs()
#trainingloop.plot_latent_space_with_labels(num_classes=n_labels, specific_classes=specific_classes)
## -------------------------

## ------ Test performance ---------
cortical_rwd_performance, striatal_rwd_performance = trainingloop.test_performance(impairCortex_afterLearning)
cortical_test_rwd_acc = np.round(sum(cortical_rwd_performance)/len(cortical_rwd_performance),decimals=2)
striatal_test_rwd_acc = np.round(sum(striatal_rwd_performance)/len(striatal_rwd_performance),decimals=2)
logging.info(f"*** | Cortical test rwd accuracy:  {cortical_test_rwd_acc*100}% | Striatal test rwd accuracy:  {striatal_test_rwd_acc*100}% | ***")
## ---------------------
