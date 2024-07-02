import torch
from IT_CNN import IT_CNN
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

class AECNN_TrainingLoop():

    def __init__(
        self,
        training_data,
        test_data,
        n_labels,
        max_label,
        cortex_ln_rate,
        cortex_bottleneck_s,
        device
    ):

        self.training_data = training_data
        self.test_data = test_data
        self.n_labels = n_labels
        self.max_label = max_label
        self.dev = device

        # Extra n. of channels, width and n. labels for the specific dataset
        data_batch, _ = next(iter(self.training_data))

        # Extract input dimension size
        n_img_channels = data_batch.size()[1] # e.g., grayscale=1, RBG=3
        img_width_s = data_batch.size()[2]
        overall_img_s = img_width_s**2 * n_img_channels

        self.IT = IT_CNN(in_channels=n_img_channels, img_size=img_width_s, ln_rate=cortex_ln_rate, n_h_units=cortex_bottleneck_s).to(self.dev)


    def train(self, ep, t_print=100):

        train_cortex_rcstr_loss = []
        t=0

        for d,l in self.training_data:

            ## TODO: AT the moment rwd=class label assuming using class_labels =[0,1] need to adapt it to any class!!!!

            # Upload data to gpu if needed
            d = d.to(self.dev)
            l = l.to(self.dev)

            ## -------- Train CCN AE -------------
            d_prediction, _ = self.IT(d)
            reconstruction_loss = self.IT.update(d_prediction, d)
            train_cortex_rcstr_loss.append(reconstruction_loss.detach())
            ## -----------------------------------
            t+=1

            
            if t % t_print == 0:
                cortex_rcstr_loss = None
                if len(train_cortex_rcstr_loss) !=0:
                    cortex_rcstr_loss = sum(train_cortex_rcstr_loss)/len(train_cortex_rcstr_loss)
                logging.info(f"| Epoch: {ep} |  Step: {t} | Cortex reconstruction loss: {cortex_rcstr_loss} |")
                train_cortex_rcstr_loss = []


    def plot_imgs(self, n_images=20):
        """ Plot n. reconstructed test images together with true test images"""

        # Extra n. of channels, width and n. labels for the specific dataset
        test_batch, _ = next(iter(self.test_data))
        color_channels = test_batch.shape[1]
        image_height = test_batch.shape[2]
        image_width = test_batch.shape[3]

        d_prediction, _ = self.IT(test_batch.to(self.dev))

        orig_images = test_batch[:n_images]
        decoded_images = d_prediction[:n_images]

        fig, axes = plt.subplots(nrows=2, ncols=n_images, sharex=True, sharey=True, figsize=(10, 2.5))

        # Loop through n. of desired images
        for i in range(n_images):
            # loop through origin and decoded images to plot the two for each n_image
            for ax, img in zip(axes, [orig_images, decoded_images]):
                curr_img = img[i].detach().to(torch.device('cpu'))
                if color_channels > 1:
                    curr_img = np.transpose(curr_img, (1, 2, 0))
                    ax[i].imshow(curr_img)
                else:
                    ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')

        plt.show()

    def plot_latent_space_with_labels(self,num_classes, specific_classes):

        d = {i:[] for i in range(num_classes)}
        test_batch, test_label_batch = next(iter(self.test_data))

        cortical_prediction, _, embedding = self.cortex(test_batch.to(self.dev))
        
        if specific_classes is not None:
            classes = specific_classes
        else:    
            classes = np.arange(0,num_classes)

        e=0
        for i in classes:
            if i in test_label_batch:
                mask = test_label_batch == i
                d[e].append(embedding[mask].detach().to('cpu').numpy())
                e+=1

        colors = list(mcolors.TABLEAU_COLORS.items())
        for i in range(num_classes):
            d[i] = np.concatenate(d[i])
            plt.scatter(
                d[i][:, 0], d[i][:, 1],
                color=colors[i][1],
                label=f'{classes[i]}',
                alpha=0.5)

        plt.legend()
        plt.show()
