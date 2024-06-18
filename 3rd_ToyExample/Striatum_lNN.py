import torch
import torch.nn as nn
import torch.optim as opt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cross_entropy


class Striatum_lNN(nn.Module):

    def __init__(self, input_s, IT_inpt_s, ln_rate, h_size, dev):
        """ Implement a Striatal network as a linear neural network, lNN, trying to predict the rwd """

        super().__init__()

        self.input_s = input_s
        self.IT_inpt_s = IT_inpt_s
        self.ET_inpt_s = 1 # provide value prediction
        self.dev = dev

        # Striatum takes input x, IT input and ET input
        self.l1 = nn.Linear(self.input_s + self.IT_inpt_s + self.ET_inpt_s ,h_size)

        self.l_rwd_output = nn.Linear(h_size, 1) 

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)

    def forward(self, x, IT_inpt, ET_inpt):
        """ Process the cortical inputs through the two separate striatal components, then unify the two to make a prediction"""

        # Concatenate input with 1st layer cortical reprs. , while reshaping in appropriate shape
        x = torch.cat([x.view(-1,self.input_s), IT_inpt.view(-1,self.IT_inpt_s), ET_inpt.view(-1,self.ET_feedback)],dim=-1)
        h = self.l1(x)

        ## .detach() to prevent rwds (dopamine) to shape striatal representations, but only readout to predict rewards
        rwd_logits = torch.sigmoid(self.l_rwd_output(h))

        return  rwd_logits

    def update(self, rwd_pred, target_rwd):
        """ update the newtork based on observed rwd"""

        loss = nn.functional.mse_loss(rwd_pred.squeeze(), target_rwd)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
