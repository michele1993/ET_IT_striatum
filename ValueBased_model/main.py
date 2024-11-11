import torch
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data
from utils import setup_logger
import logging
from TrainingLoop import TrainingLoop
import matplotlib.pyplot as plt
 
""" 
Final value-based model
"""

## ---- Set seed for reproducibility purposes
save_file = False
seeds_1 = [62419, 87745, 55327, 31023, 21716]
seeds_2 = [91207, 61790, 12391, 57053, 81513]

seeds = seeds_1 + seeds_2

# Select correct device
if torch.cuda.is_available():
    dev='cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): ## for MAC GPU usage
    dev='mps'
else:
    dev='cpu'

setup_logger()

## Experiment variables
dataset_name = "synthetic_data" #"synthetic_data" #mnist" #"cifar10"
ET_feedback = True
IT_feedback = True
striatal_h_state = 116 # model striatum as a small (linear) NN

# Select class comparison
## The main plots are based on class 0 vs 2:
class_0_comp = [[0,2],[0,1],[0,4],[0,5],[0,3],[0,6],[0,7]] # NOTE: key to maintain this order
class_1_comp = [[1,2],[1,4],[1,5],[1,3],[1,6],[1,7]] # NOTE: key to maintain this order
all_classes_comp = class_0_comp + class_1_comp
class_comp = 12
specific_classes = all_classes_comp[class_comp]


# Training variables
epocs = 20
batch_s = 50
striatal_ln_rate = 2.5e-5#3.5e-5#9e-6 #1e-5 #1e-3 #1e-5

for s in seeds:
    torch.manual_seed(s)
    np.random.seed(s)

    # Get data organised in batches 
    training_data, test_data, n_labels = get_data(dataset_name=dataset_name,batch_s=batch_s, specific_classes=specific_classes)

    # Allows to use correct rwd function depending on
    max_label = np.max(specific_classes)

    # Initialise training loop
    trainingloop = TrainingLoop(dataset_name=dataset_name, training_data=training_data, test_data=test_data, n_labels=n_labels, 
                                max_label=max_label, striatal_ln_rate=striatal_ln_rate, striatal_h_state=striatal_h_state, 
                                IT_feedback=IT_feedback, ET_feedback=ET_feedback, device=dev)

    tot_rwd_acc = []
    tot_noCortex_rwd_acc = []
    tot_CS_p_rwd = []
    tot_CS_m_rwd = []
    # Ensure baseline values are stored at trial zero
    #tot_CS_p_rwd.append(0)
    #tot_CS_m_rwd.append(0)
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
    #t = np.arange(1,len(tot_CS_p_rwd)+1)
    #plt.plot(t, tot_CS_p_rwd)
    #plt.plot(t, tot_CS_m_rwd)
    #plt.show()

    ## ------- Save files -------
    # Create directory to store results
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(file_dir,'results','data')

    data = f'VB_{dataset_name}_LickData_seed_{s}_class_{specific_classes[0]}_vs_{specific_classes[1]}'
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
