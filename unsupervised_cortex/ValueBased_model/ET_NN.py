import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np

class ET_NN(nn.Module):

    def __init__(self, IT_input_s, ln_rate, ET_h_size=10):
        """ Implement a convolutional autoencoder to mimic cortex, trained to reconstruct images """

        super().__init__()

        ## Define ET cells trying to predict values
        self.ET_layer =  nn.Sequential(
            nn.Linear(IT_input_s, ET_h_size),
            nn.ReLU(),
        )

        self.ET_output = nn.Linear(ET_h_size, 1)

        with torch.no_grad():
            self.ET_output.bias.copy_(torch.randn(1) * 0.1 - 5 *torch.ones(1)) ## Initialise lick prob close to zero 

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)#, weight_decay=1e-4)
    
    def forward(self, IT_inpt):
        """ 
        NN  pass IT (CNN) features to an (ET) NN to do reward prediction
            Args:
                stimulus_reprs: stimulus representation from IT features
            Returns: 
                rwd_pred: the prediction of the value of the input
                ET_features: the (latent) representation build to predict the value 
        """

        ET_features = self.ET_layer(IT_inpt.detach()) # detach() to prevent ET predictions shaping IT features
        rwd_pred = torch.sigmoid(self.ET_output(ET_features))

        # return last layer representation
        return rwd_pred.squeeze(), ET_features.detach()

    def update(self, rwd_pred, target_rwd):
        """ 
        update the newtork based on mean squared loss on target rwd
        """
        self.optimizer.zero_grad()

        loss = torch.mean((target_rwd - rwd_pred)**2)
        loss.backward()
        self.optimizer.step()
        return loss

