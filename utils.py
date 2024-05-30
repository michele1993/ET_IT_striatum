import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_data(dataset_name='mnist',batch_s=64):
    """ 
    Method to load datasets from Pytoch library and organised them in bacthes
    Args:
        dataset_n: each number refers to a different dataset
        batch_s: size of the batch
    """

    if dataset_name == 'mnist':
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
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
    
    else:
        raise NotImplementedError(f" {dataset_name} is an unknown dataset")

    # Use DataLoader to get data organised in random batches
    n_labels = len(training_data.classes)
    train_dataloader = DataLoader(training_data,batch_size=batch_s,shuffle=True)#, num_workers=2)
    test_dataloader = DataLoader(test_data,batch_size=batch_s,shuffle=True)#, num_workers=2)

    return train_dataloader, test_dataloader, n_labels

def conv_size(input_s, kernel_s, stride_s, padding_s, dilation_s):
    """ Method to compute output width size after a conv or maxpool2d layer (same computation for the two)"""
    return int((input_s + 2 * padding_s - dilation_s + (kernel_s-1)-1)/stride_s +1)
