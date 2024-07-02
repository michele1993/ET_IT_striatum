import torch
import torch.nn as nn
import torch.optim as opt

class Striatum_lNN(nn.Module):

    def __init__(self, input_s, IT_inpt_s, ET_inpt_s, ln_rate, h_size):
        """ Implement a Striatal network as a linear neural network, lNN, trying to predict the rwd """

        super().__init__()

        self.thalamic_input_s = input_s
        self.IT_inpt_s = IT_inpt_s
        self.ET_inpt_s = ET_inpt_s # provide value prediction

        # Striatum takes input x, IT input and ET input, initialise different synaptic weights for each
        self.W_thalamus = nn.Linear(input_s, h_size)
        self.W_IT = nn.Linear(self.IT_inpt_s, h_size)
        self.W_ET = nn.Linear(self.ET_inpt_s, h_size)

        self.l_rwd_output = nn.Linear(h_size*3, 1) 

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)

    def forward(self, thalamic_input, IT_inpt, ET_inpt):
        """ Process the cortical inputs through the two separate striatal components, then unify the two to make a prediction"""

        # Concatenate input with 1st layer cortical reprs. , while reshaping in appropriate shape
        thalamic_input = self.W_thalamus(thalamic_input.view(-1,self.thalamic_input_s))
        IT_inpt = self.W_IT(IT_inpt.view(-1,self.IT_inpt_s))
        ET_inpt = self.W_ET(ET_inpt.view(-1,self.ET_inpt_s))

        ## concatenate all striatal inputs
        x = torch.cat([thalamic_input, IT_inpt, ET_inpt],dim=-1)

        ## .detach() to prevent rwds (dopamine) to shape striatal representations, but only readout to predict rewards
        rwd_logits = torch.sigmoid(self.l_rwd_output(x))

        return  rwd_logits

    def update(self, rwd_pred, target_rwd):
        """ update the newtork based on observed rwd"""

        loss = nn.functional.mse_loss(rwd_pred.squeeze(), target_rwd.squeeze())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
