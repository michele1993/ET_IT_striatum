import torch
from IT_NN import IT_NN
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

class AENN_TrainingLoop():

    def __init__(
        self,
        training_data,
        test_data,
        n_labels,
        max_label,
        cortex_ln_rate,
        device
    ):

        self.training_data = training_data
        self.test_data = test_data
        self.n_labels = n_labels
        self.max_label = max_label
        self.dev = device

        # Extra n. of channels, width and n. labels for the specific dataset
        data_batch, _ = next(iter(self.training_data))

        # Extract input dimension size
        input_size = data_batch.size()[1] # e.g., grayscale=1, RBG=3

        self.IT = IT_NN(input_s=input_size, ln_rate=cortex_ln_rate).to(self.dev)


    def train(self, ep, t_print=10):

        train_cortex_rcstr_loss = []
        t=0

        for d,l in self.training_data:

            # Upload data to gpu if needed
            d = d.to(self.dev)
            l = l.to(self.dev)

            ## -------- Train CCN AE -------------
            d_prediction, _ = self.IT(d)
            reconstruction_loss = self.IT.update(d_prediction, d)
            train_cortex_rcstr_loss.append(reconstruction_loss.detach())
            ## -----------------------------------
            t+=1

            
            if t % t_print == 0:
                cortex_rcstr_loss = None
                if len(train_cortex_rcstr_loss) !=0:
                    cortex_rcstr_loss = sum(train_cortex_rcstr_loss)/len(train_cortex_rcstr_loss)
                logging.info(f"| Epoch: {ep} |  Step: {t} | Cortex reconstruction loss: {cortex_rcstr_loss} |")
                train_cortex_rcstr_loss = []
