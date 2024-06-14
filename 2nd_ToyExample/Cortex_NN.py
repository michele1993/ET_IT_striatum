import torch
import torch.nn as nn
import torch.optim as opt

class Cortex_NN(nn.Module):

    def __init__(self, input_s, n_labels, ln_rate, n_h_units):
        """ Implement a NN to mimic cortex, trained to predict figure labels """

        super().__init__()

        self.input_s = input_s
        self.h_units = n_h_units

        # pass the output size of the conv block to a linear layer
        # need to multiply the final layer size by itself since images have both width and height (assuming width=height)
        self.l1 = nn.Linear(input_s, self.h_units)
        self.l2 = nn.Linear(self.h_units, self.h_units)
        self.output = nn.Linear(self.h_units, n_labels)

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)
    
    def forward(self,x):
        """ Forward network pass
            Args:
                x: the input image to be classified
            Returns: 
                logits: the model predictions (logits)
                h_1: the (latent) representation at the 1st  layer (to be passed to the Striatum NN)
                h_2: the (latent) representation at the 2nd  layer (to be passed to the Striatum NN)
        """
        
        h_1 = nn.functional.relu(self.l1(x.view(-1,self.input_s)))
        h_2 = nn.functional.relu(self.l2(h_1))

        # Need to reshape to feed to linear layer as vector
        logits = self.output(h_2)

        # return last layer representation
        return logits, h_1, h_2  

    def update(self, predictions, labels):
        """ update the newtork based on cross entropy loss """
        self.optimizer.zero_grad()
        loss = nn.functional.cross_entropy(predictions,labels)
        loss.backward()
        self.optimizer.step()
        return loss

