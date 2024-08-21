import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CCN_cortex.IT_CNN import IT_CNN
from ET_NN import ET_NN
from Striatum_lNN import Striatum_lNN
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

class TrainingLoop():

    def __init__(
        self,
        dataset_name,
        training_data,
        test_data,
        n_labels,
        max_label,
        ET_ln_rate,
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
        self.IT_feedback = IT_feedback
        self.ET_feedback = ET_feedback
        self.ET_ln_rate = ET_ln_rate
        self.dev = device

        # Extra n. of channels, width and n. labels for the specific dataset
        data_batch, _ = next(iter(self.training_data))

        # Extract input dimension size
        n_img_channels = data_batch.size()[1] # e.g., grayscale=1, RBG=3
        img_width_s = data_batch.size()[2]
        overall_img_s = img_width_s**2 * n_img_channels


        # Initialise and load pretrained IT model
        self.IT = IT_CNN(in_channels=n_img_channels, img_size=img_width_s).to(self.dev)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_dir = os.path.join(file_dir,'..','models')
        IT_model_dir = os.path.join(file_dir,f'{dataset_name}_IT_model.pt')
        self.IT.load_state_dict(torch.load(IT_model_dir,  map_location=self.dev))

        # Initialise ET model and striatum
        IT_reps_s = self.IT.cnnOutput_size 
        self.ET = ET_NN(IT_input_s=IT_reps_s, ln_rate=ET_ln_rate, ET_h_size=cortex_ET_s).to(self.dev)
        self.striatum = Striatum_lNN(input_s=overall_img_s, IT_inpt_s=IT_reps_s, ln_rate=striatal_ln_rate, h_size=striatal_h_state, device=self.dev).to(self.dev)


    def train(self, ep, t_print=20):

        train_striatal_rwd_loss = []
        train_striatal_noCortex_loss = []
        train_CS_p_rwd = []
        train_CS_m_rwd = []

        tot_striatal_rwd_loss = []
        tot_striatal_noCortex_loss = []
        tot_CS_p_rwd = []
        tot_CS_m_rwd = []

        t=0
        for d,l in self.training_data:

            ## TODO: AT the moment rwd=class label assuming using class_labels =[0,1] need to adapt it to any class!!!!

            # Upload data to gpu if needed
            d = d.to(self.dev)
            l = l.to(self.dev)

            ## Get IT features
            _ , IT_features = self.IT(d)
            ## Pass a zero vector as cortical input to striatum to mimic 
            ## blocked IT cells
            if not self.IT_feedback:
                #IT_features = torch.zeros_like(IT_features).to(self.dev)
                IT_features = torch.randn_like(IT_features).to(self.dev)
        
            # Striatum selects action
            striatal_rwd_pred, noCortex_rwd_pred = self.striatum(d, IT_features)

            # Compute rwd (i.e., rwd=1 for one class, rwd=0 for the other)
            CS_rwd = torch.floor(l/self.max_label) # generalise rwd function to any two class labels

            ## ------ Compute ET prediction ------------
            ET_rwd_pred, _ = self.ET(IT_features)
            self.ET.update(ET_rwd_pred, CS_rwd)

            ## -------- Train Striatum -------------

            # find index of CS+ and CS- trials
            CS_p_indx = CS_rwd == 1
            CS_m_indx = CS_rwd == 0

            # Update CS- only if ET feedback available (i.e., assumption ET need to update for RPE<0)
            if self.ET_feedback: 
                rwd_loss, noCortex_loss = self.striatum.update(striatal_rwd_pred, noCortex_rwd_pred, CS_rwd) 
            else:        
                rwd_loss, noCortex_loss = self.striatum.update(striatal_rwd_pred[CS_p_indx], noCortex_rwd_pred[CS_p_indx], CS_rwd[CS_p_indx]) 

            train_striatal_rwd_loss.append(rwd_loss)
            train_striatal_noCortex_loss.append(noCortex_loss)

            # Store predicted value for CS+ and CS-
            train_CS_p_rwd.append(striatal_rwd_pred[CS_p_indx].detach().mean().item())
            train_CS_m_rwd.append(striatal_rwd_pred[CS_m_indx].detach().mean().item())
            ## -----------------------------------

            t+=1
            if t % t_print == 0:
                striatal_rwd_loss = None
                if len(train_striatal_rwd_loss) !=0:
                    # Accuracy
                    striatal_rwd_loss = sum(train_striatal_rwd_loss)/len(train_striatal_rwd_loss)
                    striatal_noCortex_loss = sum(train_striatal_noCortex_loss)/len(train_striatal_noCortex_loss)

                    CS_p_rwd = sum(train_CS_p_rwd)/len(train_CS_p_rwd)
                    CS_m_rwd = sum(train_CS_m_rwd)/len(train_CS_m_rwd)

                    tot_striatal_rwd_loss.append(striatal_rwd_loss)
                    tot_striatal_noCortex_loss.append(striatal_noCortex_loss)
                    tot_CS_p_rwd.append(CS_p_rwd)
                    tot_CS_m_rwd.append(CS_m_rwd)
                    
                logging.info(f"\n | Epoch: {ep} |  Step: {t} | Striatal rwd loss: {striatal_rwd_loss} | Striatal NoCortex loss: {striatal_noCortex_loss}")
                logging.info(f" CS+ rwd pred: {CS_p_rwd} | CS- rwd pred: {CS_m_rwd} | \n")
                train_striatal_rwd_loss = []
                train_striatal_noCortex_loss = []
                train_CS_p_rwd = []
                train_CS_m_rwd = []

        return tot_striatal_rwd_loss, tot_striatal_noCortex_loss, tot_CS_p_rwd, tot_CS_m_rwd


    def test_performance(self):

        actions = []
        cortexDep_rwd_performance = []
        noCortex_rwd_performance = []

        with torch.no_grad():
            for d, l in self.test_data:
                d = d.to(self.dev)
                l = l.to(self.dev)

                # Compute rwd (i.e., rwd=1 for one class, rwd=0 for the other)
                target_rwd = torch.floor(l/self.max_label) # generalise rwd function to any two class labels
                
                ## Get IT features
                _ , IT_features = self.IT(d)

                ## Pass a zero vector as cortical input to striatum to mimic blocked IT cells
                if not self.IT_feedback:
                    IT_features = torch.randn_like(IT_features).to(self.dev)

                
                cortexDep_rwd_pred, noCortex_rwd_pred = self.striatum(d, IT_features)



                ## ------ Striatum rwd test performance ----------
                cortexDep_rwd_acc = torch.sum((target_rwd == torch.round(striatal_rwd_pred.detach()))).item() /len(l)
                noCortex_rwd_acc = torch.sum((target_rwd == torch.round(noCortex_rwd_pred.detach()))).item() /len(l)
                cortexDep_rwd_performance.append(striatal_rwd_acc)
                noCortex_rwd_performance.append(noCortex_rwd_acc)
                ## ----------------------------------------------

        return cortexDep_rwd_performance, noCortex_rwd_performance

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
