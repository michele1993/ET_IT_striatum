import torch
import numpy as np
from TrainingLoop import TrainingLoop
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data
from utils import setup_logger
import logging
 
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
ET_feedback = True
IT_feedback = True
cortex_h_state = 256 # model cortex as a large (powerful) NN
striatal_h_state = 156 # model striatum as a small (linear) NN
impairCortex_afterLearning = False # At the moment assessed on test data
specific_classes = [0,1] # only ask two discriminate between two classes

# Training variables
epocs = 10
batch_s = 64
striatum_training_delay = 5 # delay training of the striatum by n. epocs, to allow cortex to learn good reprs. first
cortex_ln_rate = 1e-3
striatal_ln_rate = 1e-3

# Get data organised in batches 
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s, specific_classes=specific_classes)

# Initialise training loop
trainingloop = TrainingLoop(training_data=training_data, test_data=test_data, n_labels=n_labels, striatum_training_delay=striatum_training_delay, 
                            cortex_ln_rate=cortex_ln_rate, cortex_h_state=cortex_h_state, 
                            striatal_ln_rate=striatal_ln_rate, striatal_h_state=striatal_h_state, 
                            IT_feedback=IT_feedback, ET_feedback=ET_feedback, device=dev)

for e in range(epocs):
    trainingloop.train(e)

cortical_test_performance, striatal_test_class_performance, striatal_test_rwd_performance = trainingloop.test_performance(impairCortex_afterLearning)
cortical_test_acc = np.round(sum(cortical_test_performance)/len(cortical_test_performance),decimals=2)
striatal_test_class_acc = np.round(sum(striatal_test_class_performance)/len(striatal_test_class_performance),decimals=2)

striatal_test_rwd_acc = np.round(sum(striatal_test_rwd_performance)/len(striatal_test_rwd_performance),decimals=2)

logging.info(f"*** | Cortical test accuracy:  {cortical_test_acc*100}% | Striatal test class accuracy:  {striatal_test_class_acc*100}% | Striatal test rwd accuracy:  {striatal_test_rwd_acc*100}% | ***")
#t=0
#for d,l in training_data:
#    print(d.size())
#    print(l.size())
#    t+=1
#print(t)

