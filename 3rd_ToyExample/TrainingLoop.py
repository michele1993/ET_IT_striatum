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
        striatum_training_delay,
        cortex_ln_rate,
        cortex_h_state,
        striatal_ln_rate,
        striatal_h_state,
        IT_feedback,
        ET_feedback,
        device
    ):

        self.training_data = training_data
        self.test_data = test_data
        self.n_labels = n_labels
        self.striatum_training_delay = striatum_training_delay
        self.IT_feedback = IT_feedback
        self.dev = device

        # Extra n. of channels, width and n. labels for the specific dataset
        data_batch, _ = next(iter(self.training_data))

        # Extract input dimension size
        n_img_channels = data_batch.size()[1] # e.g., grayscale=1, RBG=3
        img_width_s = data_batch.size()[2]

        self.cortex = Cortex_CNN(in_channels=n_img_channels, img_size=img_width_s, ln_rate=cortex_ln_rate, n_h_units=cortex_h_state).to(self.dev)

        #IT_reps_s = self.cortex.cnnLayer1_size**2 * self.cortex.out_channels
        #self.striatum = Striatum_lNN(input_s=input_s, IT_inpt_s=IT_reps_s,

    def train(self, ep, t_print=100):

        train_cortex_loss = []
        train_striatal_rwd_loss = []
        t=0
        for d,l in self.training_data:

            ## TODO: AT the moment rwd=class label assuming using class_labels =[0,1] need to adapt it to any class!!!!

            # Upload data to gpu if needed
            d = d.to(self.dev)
            l = l.to(self.dev)

            ## -------- Train cortex -------------
            cortical_prediction, cortical_h1, cortical_h2 = self.cortex(d)
            # Pre-train cortex and then stop updating it
            #if ep < self.striatum_training_delay:
            loss = self.cortex.update(cortical_prediction,d)
            train_cortex_loss.append(loss.detach())
            ## -----------------------------------
            t+=1

            #striatal_cortical_input = cortical_h1
        
            ## -------- Train Striatum -------------
            #if ep >= self.striatum_training_delay:
            #    ## Pass a zero vector as cortical input to striatum to mimic 
            #    ## blocked IT cells
            #    if not self.IT_feedback:
            #        cortical_h1 = torch.zeros_like(striatal_cortical_input).to(self.dev)
            #                            
            #    if not self.ET_feedback:
            #        cortical_prediction = torch.zeros_like(cortical_prediction).to(self.dev)

            #    # Striatal predictions, the strl_class pred is for training only (i.e., trying to mimic cortical predictions)
            #    strl_rwd = self.striatum(d,striatal_cortical_input.detach(),cortical_prediction.detach()) # detach() gradient to prevent striatal gradient to change cortical predictions

            #    target_rwd = l

            #    ## Train striatum to reproduce what cortex is doing (cortex predictions should be better teaching signal than labels)
            #    rwd_loss = self.striatum.update(strl_rwd, target_rwd)
            #    train_striatal_rwd_loss.append(rwd_loss.detach())
            ## -----------------------------------

            if t % t_print == 0:
                cortex_loss = None
                if len(train_cortex_loss) !=0:
                    cortex_loss = sum(train_cortex_loss)/len(train_cortex_loss)
                striatal_rwd_loss = None
                #if len(train_striatal_rwd_loss) !=0:
                #    striatal_rwd_loss = sum(train_striatal_rwd_loss)/len(train_striatal_rwd_loss)
                logging.info(f"| Epoch: {ep} |  Step: {t} |  Cortex loss: {cortex_loss} | Striatal rwd loss: {striatal_rwd_loss}")
                train_cortex_loss = []
                train_striatal_rwd_loss = []


    def test_performance(self, impairCortex_afterLearning):

        cortical_performance = []
        striatal_class_performance = []
        striatal_rwd_performance = []
        with torch.no_grad():
            for d, l in self.test_data:
                d = d.to(self.dev)
                l = l.to(self.dev)

                # Test performance for cortex
                cortical_predictions, cortical_h1, cortical_h2 = self.cortex(d)
                cortical_predicted_labels = torch.argmax(cortical_predictions,dim=-1)
                cortical_test_acc = torch.sum(cortical_predicted_labels==l).item() / len(l)
                cortical_performance.append(cortical_test_acc)

                # Test performance for striatum
                striatal_cortical_input = cortical_h1

                # Pass a zero vector as cortical input to striatum to mimic 
                ## blocked IT cells
                if not self.IT_feedback or impairCortex_afterLearning: # since ET only needed for learning, only need to impair IT to mimic cortex damage after learning
                    cortical_h1 = torch.zeros_like(striatal_cortical_input).to(self.dev)
                # ----------
                striatal_class_predictions, striatal_rwd_prediction = self.striatum(d,striatal_cortical_input.detach())

                ## ------ Striatum classif test performance ----------
                striatal_predicted_labels = torch.argmax(striatal_class_predictions,dim=-1)
                striatal_class_acc = torch.sum(striatal_predicted_labels==l).item() / len(l)
                striatal_class_performance.append(striatal_class_acc)
                ## ---------------------------------------------------

                ## ------ Striatum rwd test performance ----------
                target_rwd = (l >= (self.n_labels//2)).int()
                striatal_rwd_acc = torch.sum((target_rwd == torch.round(striatal_rwd_prediction.squeeze()))).item() /len(l) 
                striatal_rwd_performance.append(striatal_rwd_acc)
                ## ----------------------------------------------

        return cortical_performance, striatal_class_performance, striatal_rwd_performance

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
