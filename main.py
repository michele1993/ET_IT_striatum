import torch
import numpy as np
from utils import get_data
from utils import setup_logger
from TrainingLoop import TrainingLoop
import logging
 


# Select correct device
if torch.cuda.is_available():
    dev='cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): ## for MAC GPU usage
    dev='mps'
else:
    dev='cpu'

setup_logger()
## Experiment variables
dataset_name= "mnist" #"cifar10" 

# Training variables
epocs = 1
batch_s = 64
striatum_training_delay = 0 # delay training of the striatum by n. epocs, to allow cortex to learn good reprs. first
cortex_ln_rate = 1e-3
striatal_ln_rate = 1e-3

# Get data organised in batches 
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s)

# Initialise training loop
trainingloop = TrainingLoop(training_data=training_data, test_data=test_data, n_labels=n_labels, striatum_training_delay=striatum_training_delay, 
                            device=dev, cortex_ln_rate=cortex_ln_rate, striatal_ln_rate=striatal_ln_rate)

for e in range(epocs):
    trainingloop.train(e)

cortical_test_performance, striatal_test_performance  = trainingloop.test_performance()
cortical_test_acc = np.round(sum(cortical_test_performance)/len(cortical_test_performance),decimals=2)
striatal_test_acc = np.round(sum(striatal_test_performance)/len(striatal_test_performance),decimals=2)
logging.info(f"*** | Cortical test accuracy:  {cortical_test_acc*100}% | Striatal test accuracy:  {striatal_test_acc*100}% ***")
#t=0
#for d,l in training_data:
#    print(d.size())
#    print(l.size())
#    t+=1
#print(t)

