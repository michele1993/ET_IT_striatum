import torch
from CNN_cortex import CNN_cortex
import logging

class TrainingLoop():

    def __init__(
        self,
        training_data,
        test_data,
        n_labels,
        device,
        ln_rate=1e-3
    ):

        self.training_data = training_data
        self.test_data = test_data
        self.dev = device

        # Extra n. of channels, width and n. labels for the specific dataset
        data_batch, _ = next(iter(self.training_data))
        n_input_channels = data_batch.size()[1] # e.g., grayscale=1, RBG=3
        width_s = data_batch.size()[2]

        self.cortex = CNN_cortex(n_input_channels, width_s, n_labels, ln_rate).to(self.dev)
    
    def train(self, ep, t_print=100):

        train_loss = []
        t=0
        for d,l in self.training_data:

            # Upload data to gpu if needed
            d = d.to(self.dev)
            l = l.to(self.dev)

            ## -------- Train cortex -------------
            predictions, _, _ = self.cortex(d)
            loss = self.cortex.update(predictions,l)
            train_loss.append(loss.detach())
            ## -----------------------------------
            t+=1

            if t % t_print == 0:
                loss = sum(train_loss)/len(train_loss)
                logging.info(f"| Epoch: {ep} |  Step: {t} |  Loss: {loss}")
                train_loss = []

    def test_performance(self):

        test_performance = []
        with torch.no_grad():
            for d, l in self.test_data:
                d = d.to(self.dev)
                l = l.to(self.dev)

                predictions, _,_ = self.cortex(d)
                predicted_labels = torch.argmax(predictions,dim=-1)
                test_acc = torch.sum(predicted_labels==l).item() / len(l)
                test_performance.append(test_acc)
        return test_performance
