# A circuit model of striatum and ET-IT cells interactions

The `ValueBased_model` directory contains all the files to train the value network on the cue reward association task. This can be done by running the `main.py` file in there. 
**Crucially**, before doing that, the value network requires a pre-trained IT network, which provides the stimulus features. To pre-train the IT network, you can run the `main.py` file in the `preTrain_IT_network` directory.
