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

# Get data organised in batches 
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s)

# Initialise training loop
trainingloop = TrainingLoop(training_data, test_data, n_labels, dev)

for e in range(epocs):
    trainingloop.train(e)

test_performance = trainingloop.test_performance()
test_acc = np.round(sum(test_performance)/len(test_performance),decimals=2)
logging.info(f"*** Test accuracy:  {test_acc*100}% ***")
#t=0
#for d,l in training_data:
#    print(d.size())
#    print(l.size())
#    t+=1
#print(t)

