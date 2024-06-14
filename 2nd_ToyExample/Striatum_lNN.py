import torch
import torch.nn as nn
import torch.optim as opt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cross_entropy


class Striatum_lNN(nn.Module):

    def __init__(self, input_s, cortical_input, output, ln_rate, ET_feedback, h_size, dev):
        """ Implement a Striatal network as a linear neural network, lNN, trying to predict the image labels from the images and cortical inputs """

        super().__init__()

        self.input_s = input_s
        self.cortical_input_s = cortical_input
        self.output_s = output
        self.ET_feedback = ET_feedback
        self.dev = dev

        # Implement two separate striatal components, which receives different cortial inputs, but the same sensory input
        self.l1 = nn.Linear(self.input_s + self.cortical_input_s,h_size)

        self.l_class_output = nn.Linear(h_size, output) 
        self.l_rwd_output = nn.Linear(h_size, 1) 

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)

    def forward(self, x, cortical_inpt):
        """ Process the cortical inputs through the two separate striatal components, then unify the two to make a prediction"""

        # Concatenate input with 1st layer cortical reprs. , while reshaping in appropriate shape
        x = torch.cat([x.view(-1,self.input_s),cortical_inpt.view(-1,self.cortical_input_s)],dim=-1)
        h = self.l1(x)

        ## Model ET feedback to striatum as shaping striatal representations to predict cortical outputs (i.e., classification logits)
        if self.ET_feedback:
            class_logits = self.l_class_output(h)
        else:    
            class_logits = torch.zeros((1,self.output_s)).to(self.dev)

        ## .detach() to prevent rwds (dopamine) to shape striatal representations, but only readout to predict rewards
        rwd_logits = torch.sigmoid(self.l_rwd_output(h.detach()))

        return class_logits, rwd_logits

    def update(self, class_pred_logit, label_logits, rwd_pred, target_rwd):
        """ update the newtork based on cross entropy or KL loss """

        class_pred_p = nn.functional.softmax(class_pred_logit,dim=-1)

        ## ------ Cross-entropy loss for matching distributions -----
        label_p = nn.functional.softmax(label_logits,dim=-1)
        c_entropy = cross_entropy(class_pred_p, label_p).mean()
        ## ---------------------------------------------

        ## ------ KL loss for matching distributions -----
        #log_label_p = nn.functional.log_softmax(label_logits,dim=-1)
        ## Pass target prob in log space for numerical stability
        #kl_loss = torch.nn.functional.kl_div(class_pred_p,log_label_p,log_target=True)
        ## ---------------------------------------------

        class_loss = c_entropy 

        rwd_loss = nn.functional.mse_loss(rwd_pred.squeeze(), target_rwd)
        loss = class_loss + rwd_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return class_loss, rwd_loss
