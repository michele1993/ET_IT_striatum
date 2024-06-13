import torch
import torch.nn as nn
import torch.optim as opt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cross_entropy


class Striatum_lNN(nn.Module):

    def __init__(self, img_size,in_channels, cortical_input, output, ln_rate, h_size=116):
        """ Implement a Striatal network as a linear neural network, lNN, trying to predict the image labels from the images and cortical inputs """

        super().__init__()

        self.img_overal_size = img_size*img_size*in_channels 
        self.cortical_input_s = cortical_input

        # Implement two separate striatal components, which receives different cortial inputs, but the same sensory input
        self.l1 = nn.Linear(self.img_overal_size + self.cortical_input_s,h_size)

        self.l_class_output = nn.Linear(h_size, output) 
        self.l_rwd_output = nn.Linear(h_size, 1) 

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)
    
    def forward(self, img, cortical_inpt):
        """ Process the cortical inputs through the two separate striatal components, then unify the two to make a prediction"""

        # Concatenate input image with 1st layer cortical reprs. , while reshaping in appropriate shape
        x = torch.cat([img.view(-1,self.img_overal_size),cortical_inpt.view(-1,self.cortical_input_s)],dim=-1)
        h = self.l1(x)

        class_logits = self.l_class_output(h)
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
