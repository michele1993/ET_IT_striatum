import torch
import torch.nn as nn
import torch.optim as opt

class Striatum_lNN(nn.Module):

    def __init__(self, img_size,in_channels, cortical_input_1, cortical_input_2, output, ln_rate, h_size=116):
        """ Implement a Striatal network as a linear neural network, lNN, trying to predict the image labels from the images and cortical inputs """

        super().__init__()

        self.img_overal_size = img_size*img_size*in_channels 
        self.cortical_input_1 = cortical_input_1
        self.cortical_input_2 = cortical_input_2

        # Implement two separate striatal components, which receives different cortial inputs, but the same sensory input
        self.l1 = nn.Linear(self.img_overal_size + self.cortical_input_1,h_size)
        self.l2 = nn.Linear(self.img_overal_size + self.cortical_input_2,h_size)

        self.l_output = nn.Linear(2*h_size, output) 

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)
    
    def forward(self, img, cortical_inpt_1, cortical_inpt_2):
        """ Process the cortical inputs through the two separate striatal components, then unify the two to make a prediction"""

        # Concatenate input image with 1st layer cortical reprs. , while reshaping in appropriate shape
        input_1 = torch.cat([img.view(-1,self.img_overal_size),cortical_inpt_1.view(-1,self.cortical_input_1)],dim=-1)
        h1 = self.l1(input_1)

        # Concatenate input image with 2nd layer cortical reprs. , while reshaping in appropriate shape
        input_2 = torch.cat([img.view(-1,self.img_overal_size),cortical_inpt_2.view(-1,self.cortical_input_2)],dim=-1)
        h2 = self.l2(input_2)

        h = torch.cat([h1,h2],dim=-1)

        logits = self.l_output(h)

        return logits

    def update(self, predictions, labels):
        """ update the newtork based on cross entropy loss """
        self.optimizer.zero_grad()
        loss = nn.functional.cross_entropy(predictions,labels)
        loss.backward()
        self.optimizer.step()
        return loss


