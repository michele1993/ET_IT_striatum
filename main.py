import torch
from utils import get_data
from TrainingLoop import TrainingLoop
 
## Experiment variables
dataset_name= "mnist" #"cifar10" 

# Training variables
epocs = 10
batch_s = 64

# Get data organised in batches 
training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s)

# Initialise training loop
trainingloop = TrainingLoop(training_data, n_labels)

#trainingloop.train(epocs)


t=0
for d,l in training_data:

    #print(d.size())
    #print(l.size())
    t+=1
print(t)

