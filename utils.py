import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import logging
#from sklearn.manifold import TSNE

def tsne_plotting(data, n_labels):

    # Apply t-SNE to reduce the data to 2D
    #data_2d = TSNE(n_components=2).fit_transform(data)

    # Plot the 2D representation of the data
    plt.figure(figsize=(8, 8))
    for i in range(n_labels):
        plt.scatter(\
            data_2d[labelsFull == i][:, 0],\
            data_2d[labelsFull == i][:, 1],\
            label=f"Class {i}")
    plt.legend()
    plt.title("2D representation of data using t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

def generate_20DGauss_data(n_training_samples, n_test_samples, specific_classes):
    """
    Generates synthetic 20-dimensional data
    using a mixture of Gaussians.
    """
    if specific_classes is not None:
        assert np.max(specific_classes) <8, 'Can only provide at most 8 classes to sythetic data'

    # 20-dimensional means for the Gaussians
    means = [
        np.array([2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11]),
        np.array([3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12]),
        np.array([4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13]),
        np.array([5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14]),
        np.array([6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15]),
        np.array([2, 3, 5, 6, 8, 7, 9, 10, 12, 11, 4, 5, 7, 8, 10, 9, 11, 12, 14, 13]),
        np.array([3, 2, 6, 5, 7, 8, 10, 9, 11, 12, 5, 4, 8, 7, 9, 10, 12, 11, 13, 14]),
        np.array([4, 3, 7, 6, 8, 9, 11, 10, 12, 13, 6, 5, 9, 8, 10, 11, 13, 12, 14, 15]),
    ]

    # Select specific subset of classes
    if specific_classes is not None:
        all_means = means
        means = []
        for c in specific_classes:
            means.append(all_means[c])

    # Diagonal covariances for simplicity
    cov = np.diag([3.5]*20)

    training_data = []
    training_labels = []
    test_data = []
    test_labels = []
    # Create all data one class at the time
    for i, mean in enumerate(means):
        training_samples_x_class = n_training_samples // len(means)
        test_samples_x_class = n_test_samples // len(means)
        #Generate test and training data together
        x = np.random.multivariate_normal(mean, cov, training_samples_x_class+test_samples_x_class)
        # Training data
        x_training = x[:training_samples_x_class]
        training_data.append(x_training)
        training_labels.append(np.full((x_training.shape[0]), i))
        # Test data
        x_test = x[training_samples_x_class:]
        test_data.append(x_test)
        test_labels.append(np.full((x_test.shape[0]), i))

    # Stack training data appropritately and convert to Tensor
    training_data = np.vstack(training_data)
    training_labels = np.hstack(training_labels)
    #tsne_plotting(training_data,len(means))
    training_data = torch.tensor(training_data, dtype=torch.float32)
    training_labels = torch.tensor(training_labels, dtype=torch.float32)

    # Stack test data appropritately and convert to Tensor
    test_data = np.vstack(test_data)
    test_labels = np.hstack(test_labels)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    # return training and test data in format suitable for dataloader, plus n_labels
    return list(zip(training_data, training_labels)), list(zip(test_data, test_labels)), len(means)

def get_data(dataset_name='mnist',batch_s=64, specific_classes=None):
    """ 
    Method to load datasets from Pytoch library and organised them in bacthes
    Args:
        dataset_n: each number refers to a different dataset
        batch_s: size of the batch
    """

    assert specific_classes is None or isinstance(specific_classes, list), 'specific_classes must be a list or None'

    if dataset_name == 'mnist':
        training_data = datasets.MNIST(
            root="../data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.MNIST(
            root="../data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        n_labels = len(training_data.classes)

    elif dataset_name == 'cifar10':
        training_data = datasets.CIFAR10(
            root='./data', 
            train=True,
            download=True, 
            transform=ToTensor()
        )

        test_data = datasets.CIFAR10(
            root='./data', 
            train=True,
            download=True, 
            transform=ToTensor()
        )

        n_labels = len(training_data.classes)
    
    elif dataset_name == 'synthetic_data':
        training_data, test_data, n_labels = generate_20DGauss_data(n_training_samples=800,n_test_samples=80,specific_classes=specific_classes)

    else:
        raise NotImplementedError(f" {dataset_name} is an unknown dataset")

    ## --------- Extract only data for a subset of specified classes -------
    if specific_classes is not None and dataset_name!='synthetic_data':
        # take first class in the classes list
        tot_training_indx = training_data.targets==specific_classes[0]
        tot_test_indx = test_data.targets==specific_classes[0]
        # Loop over other classes to add them to overall indx of selected classes
        for c in specific_classes[1:]:
            training_indx = training_data.targets==c
            test_indx = test_data.targets==c
            tot_training_indx = torch.logical_or(tot_training_indx,training_indx)
            tot_test_indx = torch.logical_or(tot_test_indx,test_indx)
        n_labels = len(specific_classes)
    ## ---------------------------------------------------------------------

    # Use DataLoader to get data organised in random batches
    train_dataloader = DataLoader(training_data,batch_size=batch_s,shuffle=True)#, num_workers=2)
    test_dataloader = DataLoader(test_data,batch_size=batch_s,shuffle=True)#, num_workers=2)

    return train_dataloader, test_dataloader, n_labels

def conv_size(input_s, kernel_s, stride_s, padding_s, dilation_s):
    """ Method to compute output width size after a conv or maxpool2d layer (same computation for the two)"""
    return int((input_s + 2 * padding_s - dilation_s * (kernel_s-1)-1)/stride_s +1)

def setup_logger(seed=None):
    """ set useful logger set-up"""
    logging.basicConfig(format='%(asctime)s %(message)s', encoding='utf-8', level=logging.INFO)
    logging.debug(f'Pytorch version: {torch.__version__}')
    if seed is not None:
        logging.info(f'Seed: {seed}')

def cross_entropy(pred_probs, target_probs):
    """ Compute the cross entropy between two distributions
        Args:
            pred_probs: probabilities of the prediction
            target_probs: probabilities of target 
    """

    return -1*torch.sum(target_probs * torch.log(pred_probs),dim=-1)

