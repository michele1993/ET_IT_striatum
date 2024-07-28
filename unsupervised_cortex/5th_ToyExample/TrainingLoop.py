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
        striatum_training_delay,
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


        # Initialise and load pretrained IT model
        self.IT = IT_CNN(in_channels=n_img_channels, img_size=img_width_s).to(self.dev)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_dir = os.path.join(file_dir,'..','models')
        IT_model_dir = os.path.join(file_dir,f'{dataset_name}_IT_model.pt')
        self.IT.load_state_dict(torch.load(IT_model_dir,  map_location=self.dev))

        # Initialise ET model
        IT_reps_s = self.IT.cnnOutput_size 
        self.ET = ET_NN(IT_input_s=IT_reps_s, ln_rate=ET_ln_rate).to(self.dev)
        ET_reps_s = cortex_ET_s # 1 ## for now just pass the value as input

        self.striatum = Striatum_lNN(input_s=overall_img_s, IT_inpt_s=IT_reps_s, ln_rate=striatal_ln_rate, h_size=striatal_h_state, device=self.dev).to(self.dev)

        self.mean_rwd = torch.tensor(0)

    def _impair_cortex(self):
        """ Method to turn off cortical feedback to the Striatum"""
        self.IT_feedback = False
        self.ET_feedback = False

    def train(self, ep, t_print=20):

        train_cortex_rwd_loss = []
        train_striatal_rwd_loss = []
        train_striatal_sprvsd_loss = []
        train_n_hit = []
        train_n_CR = []
        t=0

        train_CSp_action_p = []
        train_CSm_action_p = []
        tot_CSp_action_p = []
        tot_CSm_action_p = []
        tot_CSp_action_p.append(0.1)
        tot_CSm_action_p.append(0.1)

        tot_mean_rwd = []
        tot_mean_rwd.append(self.mean_rwd.clone().cpu())


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
            action, action_pred_p, target_action_p  = self.striatum(d, IT_features)

            # Compute rwd (i.e., rwd=1 for one class, rwd=0 for the other)
            CS_rwd = torch.floor(l/self.max_label) # generalise rwd function to any two class labels

            CS_p_indx = CS_rwd == 1
            CS_m_indx = CS_rwd == 0

            ## Store estimated action p for each CS
            train_CSp_action_p.append(target_action_p[CS_p_indx].mean().clone().item())
            train_CSm_action_p.append(target_action_p[CS_m_indx].mean().clone().item())

            #print("\n",target_action_p[CS_p_indx].mean())
            #print(target_action_p[CS_m_indx].mean(), "\n")

            ## -------- Compute n. hit and CR ------------
            n_hit = torch.sum(action[CS_p_indx] ==1) / torch.sum(CS_p_indx)
            train_n_hit.append(n_hit.detach().item())
            #n_miss = torch.sum(action[CS_p_indx] ==0) / torch.sum(CS_p_indx)
            #n_FA = torch.sum(action[CS_m_indx] ==1) / torch.sum(CS_m_indx)
            n_CR = torch.sum(action[CS_m_indx] ==0) / torch.sum(CS_m_indx)
            train_n_CR.append(n_CR.detach().item())


            #rwd = CS_rwd * action # rwd delivery depends on the action, lick or no lick

            RPE = CS_rwd - self.mean_rwd

            #print('\n',rwd)
            #print(RPE,'\n')

            # Update ET mean rwd prediction if have access to ET 
            if self.ET_feedback:
                #indx_a = action ==1 # update only mean rwd only when lick took place
                #self.mean_rwd += torch.mean(0.005 * RPE[indx_a]) 
                self.mean_rwd = self.mean_rwd + torch.mean(0.005 * RPE)
                #self.mean_rwd = 0.5

            RPE = RPE  * action # update only for action where lick took place

            #print("\n",RPE)
            #print(CS_rwd * action, "\n")

            ## -------- Train ET  -------------
            #ET_target = rwd.clone()
            #cortical_rwd_pred, ET_features = self.ET(IT_features, action)
            #indx = torch.logical_and(action == 1 , CS_rwd == 0)
            #ET_target[indx] = -1
            #rwd_loss = self.ET.update(cortical_rwd_pred, ET_target)
            #train_cortex_rwd_loss.append(rwd_loss.detach())
            ## -----------------------------------


            ## -------- Train Striatum -------------
            if ep >= self.striatum_training_delay:
                #RPE = rwd + cortical_rwd_pred.detach().squeeze()
                superv_loss = self.striatum.update(RPE, action_pred_p, target_action_p)
                train_striatal_sprvsd_loss.append(superv_loss)
                train_striatal_rwd_loss.append(torch.mean((CS_rwd-action)**2).item())
            ## -----------------------------------

            t+=1
            if t % t_print == 0:
                striatal_rwd_loss = None
                if len(train_striatal_rwd_loss) !=0:

                    # Accuracy
                    striatal_rwd_loss = sum(train_striatal_rwd_loss)/len(train_striatal_rwd_loss)
                    striatal_sprvsd_loss = sum(train_striatal_sprvsd_loss)/len(train_striatal_sprvsd_loss)

                    # Action p for each CS
                    CSp_action_p = sum(train_CSp_action_p)/len(train_CSp_action_p)
                    CSm_action_p = sum(train_CSm_action_p)/len(train_CSm_action_p)
                    tot_CSp_action_p.append(CSp_action_p)
                    tot_CSm_action_p.append(CSm_action_p)

                    # n. hit and CR
                    striatal_n_hit = sum(train_n_hit)/len(train_n_hit)
                    striatal_n_CR = sum(train_n_CR)/len(train_n_CR)

                    # mean rwd
                    tot_mean_rwd.append(self.mean_rwd.clone().item())

                logging.info(f"\n | Epoch: {ep} |  Step: {t} | Striatal rwd loss: {striatal_rwd_loss} | Striatal supervised loss: {striatal_sprvsd_loss}")
                logging.info(f"| n. hit: {striatal_n_hit} | n. CR: {striatal_n_CR} | Mean action: {sum(action)/len(action)} | Mean rwd: {self.mean_rwd} \n |")
                #print("Mean action: ", sum(action)/len(action))
                #print("Mean rwd: ", self.mean_rwd, '\n')
                train_cortex_rwd_loss = []
                train_striatal_rwd_loss = []
                train_striatal_sprvsd_loss = []
                train_CSp_action_p = []
                train_CSm_action_p = []

        return tot_CSp_action_p, tot_CSm_action_p, tot_mean_rwd 


    def test_performance(self, impairCortex_afterLearning=False):

        if impairCortex_afterLearning:
            self._impair_cortex()

        actions = []
        striatal_rwd_performance = []
        cortex_rwd_performance = []

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

                
                cortex_action, action_pred_p, _ = self.striatum(d, IT_features)

                action = torch.round(action_pred_p.detach()).squeeze()

                cortex_action = cortex_action.detach().squeeze()

                actions.append((sum(action)/len(action)).item())

                ## Get ET features
                #cortical_rwd_pred, ET_features = self.ET(IT_features, action)
                # Test performance for cortex
                #cortical_rwd_acc = torch.sum((target_rwd == cortical_rwd_pred.squeeze())).item() /len(l) 
                #cortical_rwd_performance.append(cortical_rwd_acc)

                ## ------ Striatum rwd test performance ----------
                striatal_rwd_acc = torch.sum((target_rwd == action)).item() /len(l) 
                cortex_rwd_acc = torch.sum((target_rwd == cortex_action)).item() /len(l) 
                striatal_rwd_performance.append(striatal_rwd_acc)
                cortex_rwd_performance.append(cortex_rwd_acc)
                ## ----------------------------------------------

        return striatal_rwd_performance, cortex_rwd_performance, actions

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
