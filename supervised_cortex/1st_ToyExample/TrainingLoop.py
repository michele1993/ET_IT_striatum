import torch
from Cortex_CNN import Cortex_CNN
from Striatum_lNN import Striatum_lNN
import logging

class TrainingLoop():

    def __init__(
        self,
        training_data,
        test_data,
        n_labels,
        striatum_training_delay,
        device,
        cortex_ln_rate=1e-3,
        striatal_ln_rate=1e-3
    ):

        self.training_data = training_data
        self.test_data = test_data
        self.striatum_training_delay = striatum_training_delay
        self.dev = device

        # Extra n. of channels, width and n. labels for the specific dataset
        data_batch, _ = next(iter(self.training_data))

        # Extract useful dimension sizes
        n_img_channels = data_batch.size()[1] # e.g., grayscale=1, RBG=3
        img_width_s = data_batch.size()[2]

        self.cortex = Cortex_CNN(in_channels=n_img_channels, img_size=img_width_s, n_labels=n_labels, ln_rate=cortex_ln_rate).to(self.dev)

        cortical_reps_1_s = self.cortex.cnnLayer1_size**2 * self.cortex.out_channels  
        cortical_reps_2_s = self.cortex.cnnLayer2_size**2 * self.cortex.out_channels  

        self.striatum = Striatum_lNN(img_size=img_width_s, in_channels=n_img_channels, 
                                     cortical_input_1=cortical_reps_1_s, cortical_input_2=cortical_reps_2_s,
                                     output=n_labels,ln_rate=striatal_ln_rate).to(self.dev)
    
    def train(self, ep, t_print=100):

        train_cortex_loss = []
        train_striatal_loss = []
        t=0
        for d,l in self.training_data:

            # Upload data to gpu if needed
            d = d.to(self.dev)
            l = l.to(self.dev)

            ## -------- Train cortex -------------
            cortical_prediction, cortical_reps_1, cortical_reps_2 = self.cortex(d)
            loss = self.cortex.update(cortical_prediction,l)
            train_cortex_loss.append(loss.detach())
            ## -----------------------------------
            t+=1
        
            ## -------- Train Striatum -------------
            if ep >= self.striatum_training_delay:
                # --- Try passing random (cortical) representations to the striatum
                #cortical_reps_1 = torch.randn_like(cortical_reps_1)
                #cortical_reps_2 = torch.randn_like(cortical_reps_2)
                # ----------

                # detach() gradient to prevent striatal gradient to change cortical predictions
                strial_prediction = self.striatum(d,cortical_reps_1.detach(), cortical_reps_2.detach())
                loss = self.striatum.update(strial_prediction,l)
                train_striatal_loss.append(loss.detach())
            ## -----------------------------------

            if t % t_print == 0:
                cortex_loss = sum(train_cortex_loss)/len(train_cortex_loss)
                striatal_loss = None
                if len(train_striatal_loss) !=0:
                    striatal_loss = sum(train_striatal_loss)/len(train_striatal_loss)
                logging.info(f"| Epoch: {ep} |  Step: {t} |  Cortex loss: {cortex_loss} |  Striatal loss: {striatal_loss}")
                train_cortex_loss = []
                train_striatal_loss = []

    def test_performance(self):

        cortical_performance = []
        striatal_performance = []
        with torch.no_grad():
            for d, l in self.test_data:
                d = d.to(self.dev)
                l = l.to(self.dev)

                # Test performance for cortex
                cortical_predictions, cortical_reps_1, cortical_reps_2 = self.cortex(d)
                cortical_predicted_labels = torch.argmax(cortical_predictions,dim=-1)
                cortical_test_acc = torch.sum(cortical_predicted_labels==l).item() / len(l)
                cortical_performance.append(cortical_test_acc)

                # Test performance for striatum
                striatal_predictions = self.striatum(d,cortical_reps_1, cortical_reps_2)
                striatal_predicted_labels = torch.argmax(striatal_predictions,dim=-1)
                striatal_test_acc = torch.sum(striatal_predicted_labels==l).item() / len(l)
                striatal_performance.append(striatal_test_acc)

        return cortical_performance, striatal_performance
