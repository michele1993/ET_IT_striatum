import torch
import torch.nn as nn
import torch.optim as opt
from torch.distributions.bernoulli import Bernoulli

class Striatum_lNN(nn.Module):

    def __init__(self, input_s, IT_inpt_s, ln_rate, h_size, thalamic_input_lr=1e-3):
        """ Implement a Striatal network as a linear neural network, lNN, trying to predict the rwd """

        super().__init__()

        self.thalamic_input_s = input_s
        self.IT_inpt_s = IT_inpt_s

        # Striatum takes input x, IT input and ET input, initialise different synaptic weights for each
        self.W_thalamus = nn.Linear(input_s, h_size)

        self.W_IT = nn.Linear(self.IT_inpt_s, h_size)

        self.l_habituation = nn.Linear(h_size, 1)

        self.l_rwd_pred = nn.Linear(h_size, 1) 

        
        with torch.no_grad():
            self.l_rwd_pred.bias.copy_(torch.randn(1) * 0.1 - 5 *torch.ones(1)) ## Initialise values close to zero 
            self.l_habituation.bias.copy_(torch.randn(1) * 0.1 - 5 *torch.ones(1)) ## Initialise values close to zero 


        # Define optimizer
        self.optimizer = opt.Adam([
        #self.optimizer = opt.SGD([
            {'params': self.W_IT.parameters()},
            {'params': self.l_rwd_pred.parameters()},
            {'params': self.W_thalamus.parameters(), 'lr':thalamic_input_lr},
            {'params': self.l_habituation.parameters(), 'lr':thalamic_input_lr}
        ], lr=ln_rate )#, momentum=0.8)


    def forward(self, thalamic_input, IT_inpt):
        """ Process the cortical inputs through the two separate striatal components, then unify the two to make a prediction"""

        thalamic_input = torch.relu(self.W_thalamus(thalamic_input.view(-1,self.thalamic_input_s)))

        IT_inpt = torch.relu(self.W_IT(IT_inpt.view(-1,self.IT_inpt_s)))

        rwd_pred = torch.sigmoid(self.l_rwd_pred(IT_inpt))
        noCortex_rwd_pred = torch.sigmoid(self.l_habituation(thalamic_input))

        return  rwd_pred.squeeze(), noCortex_rwd_pred.squeeze()

    def update(self, rwd_pred, noCortex_rwd_pred, RPE_grad):
        """ update the newtork based on observed rwd"""

        #torch.sum(rwd_pred,dim=0,keepdim=True).unsqueeze(0).backward(gradient=RPE.unsqueeze(0)) 
        batch_rwd_pred = torch.sum(rwd_pred) # use sum since they are all independent batches
        rwd_pred.backward(gradient=RPE_grad)
        no_cortex_loss = torch.mean((rwd_pred.detach() - noCortex_rwd_pred)**2) # Striatum tries to learn its own predictions from contrical inputs using the thalamic inputs
        no_cortex_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return no_cortex_loss.detach().item()

