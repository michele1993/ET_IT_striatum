import torch
import torch.nn as nn
import torch.optim as opt
from utils import conv_size

class CNN_cortex(nn.Module):

    def __init__(self, in_channels, img_size, n_labels, ln_rate, out_channels=32, kernel_size=4, stride_s=1, padding_s=0, dilation_s=1, n_h_units=56):

        super().__init__()

        self.out_channels = out_channels

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

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels*2,
                kernel_size=kernel_size,
                stride=stride_s,
                padding=padding_s,
                dilation=dilation_s
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        # call all the conv & maxPool operations on an empty tensor to infer the output size of the conv block
        #n_channels = self.conv3(self.conv2(self.conv1((torch.empty(1, in_channels, img_size, img_size))))).size(-1)

        # Call method to compute output width size after a conv and a max pool operation (i.e., each conv and maxpool operation shrinks the image size)
        # NOTE: call twice (recursively) since have both conv and maxpool layer for each layer
        self.cnnLayer1_size = conv_size(conv_size(img_size,kernel_size,stride_s,padding_s,dilation_s),kernel_size,stride_s,padding_s,dilation_s) 
        self.cnnLayer2_size = conv_size(conv_size(self.cnnLayer1_size,kernel_size,stride_s,padding_s,dilation_s),kernel_size,stride_s,padding_s,dilation_s)
        self.cnnLayer3_size = conv_size(conv_size(self.cnnLayer2_size,kernel_size,stride_s,padding_s,dilation_s),kernel_size,stride_s,padding_s,dilation_s)

        # pass the output size of the conv block to a linear layer
        # need to multiply the final layer size by itself since images have both width and height (assuming width=height)
        self.linear_1 = nn.Linear(self.cnnLayer3_size * self.cnnLayer3_size * 2 * out_channels, n_labels)

        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)
    
    def forward(self,x):

        h_1 = self.conv1(x)
        h_2 = self.conv2(h_1)
        h_3 = self.conv3(h_2)

        logit = self.linear_1(h_3)

        predictions = nn.functional.softmax(logit)

        return predictions, h_1, h_2, h_3

    def update(self, predictions, labels):
        self.optimizer.zero_grad()
        loss = nn.functional.nll(predictions,labels)
        loss.backward()
        self.optimizer.step()
        return loss

