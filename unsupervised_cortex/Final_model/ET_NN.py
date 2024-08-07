import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np

class ET_NN(nn.Module):

    def __init__(self, IT_input_s, ln_rate, action_s=0, ET_h_size=10, n_h_units=512):
        """ Implement a convolutional autoencoder to mimic cortex, trained to reconstruct images """

        super().__init__()

        ## Define ET cells trying to predict values
        self.ET_layer =  nn.Sequential(
            nn.Linear(IT_input_s+action_s, n_h_units),
            nn.ReLU(),
            #nn.Dropout(0.25),
            #nn.Linear(n_h_units, n_h_units),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(n_h_units, ET_h_size),
            nn.ReLU(),
        )

        self.ET_output = nn.Linear(ET_h_size, 1)

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)#, weight_decay=1e-4)
    
    def forward(self, stimulus_reprs, a):
        """ 
        NN  pass IT (CNN) features to an (ET) NN to do reward prediction
            Args:
                stimulus_reprs: stimulus representation from IT features
                a: selected action
            Returns: 
                rwd_pred: the prediction of the value of the input
                ET_features: the (latent) representation build to predict the value 
        """

        # convert actions to one-hot
        a = nn.functional.one_hot(a.squeeze())
        #x = torch.cat([stimulus_reprs, a],dim=-1)
        x = stimulus_reprs
        
        ET_features = self.ET_layer(x.detach()) # detach() to prevent ET predictions shaping IT features
        rwd_pred = torch.tanh(self.ET_output(ET_features))

        # return last layer representation
        return rwd_pred, ET_features.detach()

    def update(self, rwd_pred, target_rwd):
        """ 
        update the newtork based on mean squared loss on target rwd
        """
        self.optimizer.zero_grad()

        rwd_loss = nn.functional.mse_loss(rwd_pred.squeeze(),target_rwd)
        rwd_loss.backward()
        self.optimizer.step()
        return rwd_loss

