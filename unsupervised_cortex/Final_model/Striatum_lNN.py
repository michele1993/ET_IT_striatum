import torch
import torch.nn as nn
import torch.optim as opt
from torch.distributions.bernoulli import Bernoulli

class Striatum_lNN(nn.Module):

    def __init__(self, input_s, IT_inpt_s, ln_rate, h_size, device):
        """ Implement a Striatal network as a linear neural network, lNN, trying to predict the rwd """

        super().__init__()

        self.thalamic_input_s = input_s
        self.IT_inpt_s = IT_inpt_s
        self.dev = device

        # Striatum takes input x, IT input and ET input, initialise different synaptic weights for each
        self.W_thalamus = nn.Linear(input_s, 5*h_size)

        self.W_IT = nn.Linear(self.IT_inpt_s, h_size)

        self.l_habituation = nn.Linear(5*h_size, 1)

        self.l_action_p = nn.Linear(h_size, 1) 
        self.l_rwd_pred = nn.Linear(h_size, 1) 

        
        with torch.no_grad():
            self.l_action_p.bias.copy_(torch.randn(1) * 0.1 - 2 *torch.ones(1)) ## Initialise lick prob close to zero 
            self.l_rwd_pred.bias.copy_(torch.randn(1) * 0.1 - 3 *torch.ones(1)) ## Initialise values close to zero 

        #self.l_rwd_output = nn.Linear(h_size, 1) 
        #self.l_rwd_output = nn.Linear(h_size+IT_inpt_s, 1) 

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)

        #self.apply(self.small_weight_init)


    def small_weight_init(self,l):
        if isinstance(l,nn.Linear):
            #nn.init.normal_(l.weight,mean=0,std= 0.000005)# std= 0.00005
            nn.init.normal_(l.bias,mean=-1,std= 0.1)#  std= 0.00005

    def forward(self, thalamic_input, IT_inpt):
        """ Process the cortical inputs through the two separate striatal components, then unify the two to make a prediction"""

        # Concatenate input with 1st layer cortical reprs. , while reshaping in appropriate shape
        #thalamic_input_1 = torch.relu(self.W_thalamus_1(thalamic_input.view(-1,self.thalamic_input_s)))
        thalamic_input = torch.relu(self.W_thalamus(thalamic_input.view(-1,self.thalamic_input_s)))

        IT_inpt = torch.relu(self.W_IT(IT_inpt.view(-1,self.IT_inpt_s)))

        target_action_p = torch.sigmoid(self.l_action_p(IT_inpt))
        rwd_pred = torch.sigmoid(self.l_rwd_pred(IT_inpt))
        noCortex_action_p = torch.sigmoid(self.l_habituation(thalamic_input))

        ## --- Initialise distribution to sample from -----
        d = Bernoulli(target_action_p)
        action = d.sample()
        self.p_action = d.log_prob(action)

        return  action.to(torch.int64).squeeze(), rwd_pred.squeeze(), noCortex_action_p.squeeze(), target_action_p.detach().squeeze() # convert action to int (i.e., lick or no lick)

    def update(self, RPE, action_pred_p, target_action_p, rwd_pred, rwd):
        """ update the newtork based on observed rwd"""

        loss_1 = torch.sum(-self.p_action.squeeze() * RPE)
        loss_2 = torch.mean((rwd - rwd_pred)**2)
        loss_3 = torch.mean((action_pred_p - target_action_p)**2)

        loss = loss_1 + 5 * loss_2 + loss_3


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_2

