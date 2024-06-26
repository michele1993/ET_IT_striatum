import torch
from Cortex_CNN import Cortex_CNN
from Striatum_lNN import Striatum_lNN
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

class TrainingLoop():

    def __init__(
        self,
        training_data,
        test_data,
        n_labels,
        max_label,
        striatum_training_delay,
        cortex_ln_rate,
        cortex_bottleneck_s,
        cortex_ET_s,
        striatal_ln_rate,
        striatal_h_state,
        IT_feedback,
        ET_feedback,
        device
    ):

        self.training_data = training_data
        self.test_data = test_data
        self.n_labels = n_labels
        self.max_label = max_label
        self.striatum_training_delay = striatum_training_delay
        self.IT_feedback = IT_feedback
        self.ET_feedback = ET_feedback
        self.dev = device

        # Extra n. of channels, width and n. labels for the specific dataset
        data_batch, _ = next(iter(self.training_data))

        # Extract input dimension size
        n_img_channels = data_batch.size()[1] # e.g., grayscale=1, RBG=3
        img_width_s = data_batch.size()[2]
        overall_img_s = img_width_s**2 * n_img_channels

        self.cortex = Cortex_CNN(in_channels=n_img_channels, img_size=img_width_s, ln_rate=cortex_ln_rate, n_h_units=cortex_bottleneck_s, ET_h_size=cortex_ET_s).to(self.dev)

        IT_reps_s = self.cortex.cnnOutput_size 
        ET_reps_s = 1 # for now just pass the value as input

        self.striatum = Striatum_lNN(input_s=overall_img_s, IT_inpt_s=IT_reps_s, ET_inpt_s= ET_reps_s, ln_rate=striatal_ln_rate, h_size=striatal_h_state).to(self.dev)

    def _impair_cortex(self):
        """ Method to turn off cortical feedback to the Striatum"""
        self.IT_feedback = False
        self.ET_feedback = False

    def train(self, ep, t_print=100):

        train_cortex_rcstr_loss = []
        train_cortex_rwd_loss = []
        train_striatal_rwd_loss = []
        t=0

        for d,l in self.training_data:

            ## TODO: AT the moment rwd=class label assuming using class_labels =[0,1] need to adapt it to any class!!!!

            # Upload data to gpu if needed
            d = d.to(self.dev)
            l = l.to(self.dev)

            # Compute rwd (i.e., rwd=1 for one class, rwd=0 for the other)
            target_rwd = torch.floor(l/self.max_label) # generalise rwd function to any two class labels

            ## -------- Train cortex -------------
            d_prediction, cortical_rwd_pred, IT_features, _ = self.cortex(d)
            # Pre-train cortex and then stop updating it
            #if ep < self.striatum_training_delay:
            reconstruction_loss, rwd_loss = self.cortex.update(d_prediction, d, cortical_rwd_pred, target_rwd)
            train_cortex_rcstr_loss.append(reconstruction_loss.detach())
            train_cortex_rwd_loss.append(rwd_loss.detach())
            ## -----------------------------------
            t+=1

        
            ## -------- Train Striatum -------------
            if ep >= self.striatum_training_delay:

                # Pass actual rwd OR the cortical pred as target to train Striatum
                striatum_target = cortical_rwd_pred.detach() # target_rwd

                ## Pass a zero vector as cortical input to striatum to mimic 
                ## blocked IT cells
                if not self.IT_feedback:
                    IT_features = torch.zeros_like(IT_features).to(self.dev)
                                        
                if not self.ET_feedback: # Assume the ET feedback conveys the rwd prediction to the striatum
                    cortical_rwd_pred = torch.zeros_like(cortical_rwd_pred).to(self.dev)

                strl_rwd = self.striatum(d, IT_features, cortical_rwd_pred.detach())
                rwd_loss = self.striatum.update(strl_rwd, striatum_target)
                train_striatal_rwd_loss.append(rwd_loss.detach())
            ## -----------------------------------

            if t % t_print == 0:
                cortex_loss = None
                if len(train_cortex_rcstr_loss) !=0:
                    cortex_rcstr_loss = sum(train_cortex_rcstr_loss)/len(train_cortex_rcstr_loss)
                    cortex_rwd_loss = sum(train_cortex_rwd_loss)/len(train_cortex_rwd_loss)
                striatal_rwd_loss = None
                if len(train_striatal_rwd_loss) !=0:
                    striatal_rwd_loss = sum(train_striatal_rwd_loss)/len(train_striatal_rwd_loss)
                logging.info(f"| Epoch: {ep} |  Step: {t} | Cortex reconstruction loss: {cortex_rcstr_loss} | Cortex rwd loss: {cortex_rwd_loss} | Striatal rwd loss: {striatal_rwd_loss}")
                train_cortex_rcstr_loss = []
                train_cortex_rwd_loss = []
                train_striatal_rwd_loss = []


    def test_performance(self, impairCortex_afterLearning=False):

        if impairCortex_afterLearning:
            self._impair_cortex()

        cortical_rwd_performance = []
        striatal_rwd_performance = []

        with torch.no_grad():
            for d, l in self.test_data:
                d = d.to(self.dev)
                l = l.to(self.dev)

                # Compute rwd (i.e., rwd=1 for one class, rwd=0 for the other)
                target_rwd = torch.floor(l/self.max_label) # generalise rwd function to any two class labels

                # Test performance for cortex
                d_prediction, cortical_rwd_pred, IT_features, _ = self.cortex(d)
                cortical_rwd_acc = torch.sum((target_rwd == torch.round(cortical_rwd_pred.squeeze()))).item() /len(l) 
                cortical_rwd_performance.append(cortical_rwd_acc)

                if not self.IT_feedback:
                    IT_features = torch.zeros_like(IT_features).to(self.dev)
                                        
                if not self.ET_feedback:
                    cortical_rwd_pred = torch.zeros_like(cortical_rwd_pred).to(self.dev)

                striatal_rwd_prediction = self.striatum(d, IT_features, cortical_rwd_pred)

                ## ------ Striatum rwd test performance ----------
                striatal_rwd_acc = torch.sum((target_rwd == torch.round(striatal_rwd_prediction.squeeze()))).item() /len(l) 
                striatal_rwd_performance.append(striatal_rwd_acc)
                ## ----------------------------------------------

        return cortical_rwd_performance, striatal_rwd_performance

    def plot_imgs(self, n_images=15):
        """ Plot n. reconstructed test images together with true test images"""

        # Extra n. of channels, width and n. labels for the specific dataset
        test_batch, _ = next(iter(self.test_data))
        color_channels = test_batch.shape[1]
        image_height = test_batch.shape[2]
        image_width = test_batch.shape[3]

        cortical_prediction, _, _ = self.cortex(test_batch.to(self.dev))

        orig_images = test_batch[:n_images]
        decoded_images = cortical_prediction[:n_images]

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
