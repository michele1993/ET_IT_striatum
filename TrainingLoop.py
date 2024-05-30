import torch
from CNN_cortex import CNN_cortex

class TrainingLoop():

    def __init__(
        self,
        training_data,
        n_labels,
        ln_rate=1e-3
    ):

        self.training_data = training_data

        # Extra n. of channels, width and n. labels for the specific dataset
        data_batch, _ = next(iter(self.training_data))
        n_channels = data_batch.size()[1]
        width_s = data_batch.size()[2]

        self.cortex = CNN_cortex(n_channels, width_s, n_labels, ln_rate)
    
    def train(self, epocs, t_print=100):

        for e in range(epocs):

            train_loss = []
            t=0
            for d,l in self.training_data:
                predicted_labels, _ = self.cortex(d)
                loss = self.cortex.update(predicted_labels,l)
                train_loss.append(loss.detach())
                t+=1

                if t % t_print == 0:
                    print("Epoch: ", e, " Step: ", t, " Loss: ", sum(train_loss)/len(train_loss))
                    train_loss = []


