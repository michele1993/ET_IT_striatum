import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np

## Define useful class to perform reshape operation inside nn.Sequential

class Reshape(nn.Module):
    def __init__(self,*args):
        super().__init__()
        self.shape = args
    def forward(self,x):
            return x.view(self.shape)

class Cortex_CNN(nn.Module):

    def __init__(self, in_channels, img_size, ln_rate, out_channels=32, kernel_size=5, stride_s=1, padding_s=0, dilation_s=1, n_h_units=56):
        """ Implement a convolutional autoencoder to mimic cortex, trained to reconstruct images """

        super().__init__()

        # First CNN layer with Maxpooling
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride_s,
                padding=padding_s,
                dilation=dilation_s
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=2*out_channels,
                kernel_size=kernel_size,
                stride=stride_s,
                padding=padding_s,
                dilation=dilation_s
            ),
            nn.ReLU(),
            nn.Flatten()
        )

        # Call all the conv & maxPool operations on an empty tensor to infer the output representation size after each conv block
        # NOTE: this is needed because each con2d and MaxPool2d operation shrinks the size of the image
        self.cnnOutput_size = self.cnn_encoder(torch.ones(1, in_channels, img_size, img_size)).size(-1) 

        # compute the output of the width/height of image coming out CNN encoder
        cnn_img_size = int(np.sqrt(self.cnnOutput_size/(2*out_channels)))

        # pass the output size of the conv block to a linear layer
        # need to multiply the final layer size by itself since images have both width and height (assuming width=height)
        self.l1 = nn.Linear(self.cnnOutput_size, n_h_units)

        self.cnn_decoder = nn.Sequential(
            nn.Linear(n_h_units, self.cnnOutput_size), # 'Revert' linear operation performed to map onto bottleneck latent space
            Reshape(-1,2*out_channels, cnn_img_size,cnn_img_size), # See helper class above!
            nn.ConvTranspose2d(
                in_channels=2*out_channels, # since passing latent state from bottleneck
                out_channels=out_channels, # trying to reconstruct the image,
                kernel_size=kernel_size,
                stride=stride_s,
                padding=padding_s,
                dilation=dilation_s
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=out_channels, # since passing latent state from bottleneck
                out_channels=in_channels, # trying to reconstruct the image,
                kernel_size=kernel_size,
                stride=stride_s,
                padding=padding_s,
                dilation=dilation_s
            ),
            nn.Sigmoid() # By default pixel images in MNIST and CIFAR10 are in range [0,1]
        )

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)
    
    def forward(self,x):
        """ CNN autoencoder pass
            Args:
                x: the input image to be reconstructed
            Returns: 
                x_hat: reconstructed image
                cnn_h: the (latent) representation build by the CCN encoder (Not the bottleneck)
                h: the (latent) representation build by the bottleneck
        """
        
        cnn_h = self.cnn_encoder(x)

        h = self.l1(cnn_h)

        x_hat = self.cnn_decoder(h)

        # return last layer representation
        return x_hat, cnn_h, h

    def update(self, predictions, targets):
        """ 
        update the newtork based on mean squared loss on target images
        Args:
            predictions: (predicted) reconstructed images
            targets: target images
        """
        self.optimizer.zero_grad()
        # Train auto-encoder with MSE
        loss = nn.functional.mse_loss(predictions,targets)
        loss.backward()
        self.optimizer.step()
        return loss

