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

class IT_CNN(nn.Module):

    def __init__(self, in_channels, img_size, ln_rate=1e-4, out_channels=16, kernel_size=3, stride_s=1, padding_s=1, dilation_s=1, bottleneck_s=56):
        """ Implement a convolutional autoencoder to mimic cortex, trained to reconstruct images """

        super().__init__()

        self.bottleneck_s = bottleneck_s

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
            nn.Conv2d(
                in_channels=2*out_channels,
                out_channels=3*out_channels,
                kernel_size=kernel_size,
                stride=stride_s,
                padding=padding_s,
                dilation=dilation_s
            ),
            nn.ReLU(),
            nn.Flatten()
        )

        # Call all the conv operations on an empty tensor to infer the output representation size after each conv block
        self.cnnOutput_size = self.cnn_encoder(torch.ones(1, in_channels, img_size, img_size)).size(-1) # use -1 since hase nn.Flatten()

        # infer the output of the width/height of image coming out CNN encoder from overall size
        cnn_img_size = int(np.sqrt(self.cnnOutput_size/(3*out_channels)))

        # pass the output size of the conv block to a linear layer
        self.l1 = nn.Linear(self.cnnOutput_size, bottleneck_s) # bottleneck

        self.cnn_decoder = nn.Sequential(
            nn.Linear(bottleneck_s, self.cnnOutput_size), # 'Revert' linear operation performed to map onto bottleneck latent space
            Reshape(-1,3*out_channels, cnn_img_size,cnn_img_size), # See helper class above!
            nn.ConvTranspose2d(
                in_channels=3*out_channels, # since passing latent state from bottleneck
                out_channels=2*out_channels, # trying to reconstruct the image,
                kernel_size=kernel_size,
                stride=stride_s,
                padding=padding_s,
                dilation=dilation_s
            ),
            nn.ReLU(),
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
        """ 
        CNN encoder modelling IT cells, pass input image through bottleneck for reconstruction
            (i.e., IT features are not trained through unsupervised learning)
            Args:
                x: the input image to be reconstructed
            Returns: 
                x_pred: reconstructed input (e.g., image)
                IT_features: the (latent) representation build by the CCN encoder by unsupervised learning (Not the bottleneck) 
        """
        
        IT_features = self.cnn_encoder(x)

        ## -------- Unsupervised learning (to train IT features only) ----------
        bottleneck = self.l1(IT_features)
        x_pred = self.cnn_decoder(bottleneck)

        # return last layer representation
        return x_pred, IT_features.detach() #bottleneck.detach()

    def update(self, x_predictions, x_targets):
        """ 
        update the newtork based on mean squared loss on target images
        Args:
            predictions: (predicted) reconstructed images
            targets: target images
        """
        self.optimizer.zero_grad()

        # Train auto-encoder with MSE
        reconstruction_loss = nn.functional.mse_loss(x_predictions,x_targets)
        loss = reconstruction_loss 
        loss.backward()
        self.optimizer.step()
        return reconstruction_loss

