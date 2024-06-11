import torch
import torch.nn as nn
import torch.optim as opt

class Cortex_CNN(nn.Module):

    def __init__(self, in_channels, img_size, n_labels, ln_rate, out_channels=32, kernel_size=5, stride_s=1, padding_s=0, dilation_s=1, n_h_units=56):
        """ Implement a CNN to mimic visual cortex, trained to predict figure labels """

        super().__init__()

        self.out_channels = out_channels


        # First CNN layer with Maxpooling
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride_s,
                padding=padding_s,
                dilation=dilation_s
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        # Second CNN layer with Maxpooling
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride_s,
                padding=padding_s,
                dilation=dilation_s
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        # Call all the conv & maxPool operations on an empty tensor to infer the output representation size after each conv block
        # NOTE: this is needed because each con2d and MaxPool2d operation shrinks the size of the image
        self.cnnLayer1_size = self.conv1(torch.empty(1, in_channels, img_size, img_size)).size(-1)
        self.cnnLayer2_size = self.conv2(torch.empty(1, out_channels, self.cnnLayer1_size, self.cnnLayer1_size)).size(-1)

        # pass the output size of the conv block to a linear layer
        # need to multiply the final layer size by itself since images have both width and height (assuming width=height)
        self.linear_1 = nn.Linear(self.cnnLayer2_size * self.cnnLayer2_size * self.out_channels, n_labels)

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)
    
    def forward(self,x):
        """ Forward network pass
            Args:
                x: the input image to be classified
            Returns: 
                logits: the model predictions (logits)
                h_1: the (latent) representation at the 1st cnn layer (to be passed to the Striatum NN)
                h_2: the (latent) representation at the 2nd cnn layer (to be passed to the Striatum NN)
        """
                    

        h_1 = self.conv1(x)
        h_2 = self.conv2(h_1)

        # Need to reshape to feed to linear layer as vector
        logits = self.linear_1(h_2.view(-1,self.cnnLayer2_size * self.cnnLayer2_size * self.out_channels))

        return logits, h_1, h_2

    def update(self, predictions, labels):
        """ update the newtork based on cross entropy loss """
        self.optimizer.zero_grad()
        loss = nn.functional.cross_entropy(predictions,labels)
        loss.backward()
        self.optimizer.step()
        return loss

