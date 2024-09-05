import torch
import numpy as np
from CNN_AE_TrainingLoop import AECNN_TrainingLoop
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import get_data
from utils import setup_logger
import logging
 
""" 
In this Toy model, I model IT as CNN features of an encoder trained with unsuperivsed learning (by including a bottleneck and decoder for training IT features only), 
, while I assume ET to build rwd predicting representations (i.e., a small layer) from the IT representations. 
The ET (latent) representations are then passed to the striatum as target for training the striatum latent representations, based on which rwds are then predicted.
"""

## ---- Set seed for reproducibility purposes
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
save_file = False

# Select correct device
if torch.cuda.is_available():
    dev='cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): ## for MAC GPU usage
    dev='mps'
else:
    dev='cpu'

setup_logger()

## Experiment variables
dataset_name = "mnist" #"synthetic_data" #mnist" #"cifar10"
specific_classes = None #[0,1] # only ask two discriminate between two classes

# Training variables
epocs = 20#0#00
batch_s = 64
cortex_ln_rate = 5e-4


# Get data organised in batches 
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s, specific_classes=specific_classes)

# Allows to use correct rwd function depending on
max_label = np.max(specific_classes)

# Initialise training loop
trainingloop = AECNN_TrainingLoop(training_data=training_data, test_data=training_data, n_labels=n_labels, max_label=max_label, 
                            cortex_ln_rate=cortex_ln_rate, device=dev) 
                            
for e in range(epocs):
    trainingloop.train(e)

## ---- Save model ------
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'..','models')

# Create directory if it did't exist before
os.makedirs(file_dir, exist_ok=True)
model_dir = os.path.join(file_dir,f'{dataset_name}_IT_model.pt')
if save_file:
    torch.save(trainingloop.IT.state_dict(),model_dir)

## ----- Useful plots ------
trainingloop.plot_imgs()
#trainingloop.plot_latent_space_with_labels(num_classes=n_labels, specific_classes=specific_classes)
## -------------------------
