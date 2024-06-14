import torch
from Cortex_NN import Cortex_NN
from Striatum_lNN import Striatum_lNN
import logging

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
        batch_s = data_batch.size()[0]
        input_s = data_batch.view(batch_s,-1).size()[1]

        self.cortex = Cortex_NN(input_s=input_s, n_labels=self.n_labels, ln_rate=cortex_ln_rate, n_h_units=cortex_h_state).to(self.dev)

        cortical_reps_s = self.cortex.h_units
        self.striatum = Striatum_lNN(input_s=input_s, cortical_input=cortical_reps_s, output=self.n_labels, ln_rate=striatal_ln_rate,
                                     ET_feedback=ET_feedback, h_size=striatal_h_state, dev=self.dev).to(self.dev)

    def train(self, ep, t_print=100):

        train_cortex_loss = []
        train_striatal_class_loss = []
        train_striatal_rwd_loss = []
        t=0
        for d,l in self.training_data:

            # Upload data to gpu if needed
            d = d.to(self.dev)
            l = l.to(self.dev)

            ## -------- Train cortex -------------
            cortical_prediction, cortical_h1, cortical_h2 = self.cortex(d)
            # Pre-train cortex and then stop updating it
            if ep < self.striatum_training_delay:
                loss = self.cortex.update(cortical_prediction,l)
                train_cortex_loss.append(loss.detach())
            ## -----------------------------------
            t+=1

            striatal_cortical_input = cortical_h1
        
            ## -------- Train Striatum -------------
            if ep >= self.striatum_training_delay:
                ## Pass a zero vector as cortical input to striatum to mimic 
                ## blocked IT cells
                if not self.IT_feedback:
                    cortical_h1 = torch.zeros_like(striatal_cortical_input).to(self.dev)

                # Striatal predictions, the strl_class pred is for training only (i.e., trying to mimic cortical predictions)
                strl_class, strl_rwd = self.striatum(d,striatal_cortical_input.detach()) # detach() gradient to prevent striatal gradient to change cortical predictions

                # For simplicity rwd=1 for second half of classes and rwd=0 for the other half of classes
                # resulting in a binary reward-based task
                target_rwd = (l >= (self.n_labels//2)).float()

                ## Train striatum to reproduce what cortex is doing (cortex predictions should be better teaching signal than labels)
                class_loss, rwd_loss = self.striatum.update(strl_class, cortical_prediction.detach(), strl_rwd, target_rwd)
                train_striatal_class_loss.append(class_loss.detach())
                train_striatal_rwd_loss.append(rwd_loss.detach())
            ## -----------------------------------

            if t % t_print == 0:
                cortex_loss = None
                if len(train_cortex_loss) !=0:
                    cortex_loss = sum(train_cortex_loss)/len(train_cortex_loss)
                striatal_class_loss = None
                striatal_rwd_loss = None
                if len(train_striatal_class_loss) !=0:
                    striatal_class_loss = sum(train_striatal_class_loss)/len(train_striatal_class_loss)
                    striatal_rwd_loss = sum(train_striatal_rwd_loss)/len(train_striatal_rwd_loss)
                logging.info(f"| Epoch: {ep} |  Step: {t} |  Cortex loss: {cortex_loss} |  Striatal class loss: {striatal_class_loss} | Striatal rwd loss: {striatal_rwd_loss}")
                train_cortex_loss = []
                train_striatal_class_loss = []
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
