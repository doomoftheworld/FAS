from common_imports import *

def get_cifar10_dataset(normalize=False):
    # Define the needed transformations
    required_transform = None
    if normalize:
        # The default normalization is mean=0.5 and std=0.5 for each channel
        required_transform = [ transforms.ToTensor(),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomErasing(), 
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    else:
        required_transform = [ transforms.ToTensor(), 
                              transforms.RandomHorizontalFlip(), 
                              transforms.RandomCrop(32, padding=4),
                              transforms.RandomErasing()
                            ]

    train_dataset= datasets.CIFAR10(
            "./data/",
            train=True,
            download=True,
            transform=transforms.Compose(required_transform),
        )

    test_dataset= datasets.CIFAR10(
            "./data/",
            train=False,
            download=True,
            transform=transforms.Compose([ transforms.ToTensor()]),
        )
    
    return train_dataset, test_dataset

def get_cifar10_dataset_without_transform():

    train_dataset= datasets.CIFAR10(
            "./data/",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    test_dataset= datasets.CIFAR10(
            "./data/",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    
    return train_dataset, test_dataset


def torch_random_split(dataset, split_portion=0.2):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)*split_portion), int(len(dataset)*split_portion)])
    return train_dataset, test_dataset