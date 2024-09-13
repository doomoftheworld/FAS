"""
This file contains the functions for the novelty data preparation.
"""
import os
from common_imports import transforms, datasets

"""
The functions for loading the OMS datasets
"""
# Functions to get the svhn dataset (as an OOD dataset)
def get_svhn_dataset_without_transform():

    train_dataset= datasets.SVHN(
            "./data/",
            split='train',
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    test_dataset= datasets.SVHN(
            "./data/",
            split='test',
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    
    return train_dataset, test_dataset

# Functions to get the dtd dataset (as an OOD dataset)
def get_dtd_dataset_resized():

    train_dataset= datasets.DTD(
            "./data/",
            split='train',
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]),
        )

    test_dataset= datasets.DTD(
            "./data/",
            split='test',
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]),
        )
    
    return train_dataset, test_dataset

# Functions to get the places365 dataset (as an OOD dataset)
def get_places_test_dataset_resized():

    test_dataset = None
    if os.path.exists("./data/val_256"):
        test_dataset= datasets.Places365(
                "./data/",
                split='val',
                download=False,
                small=True, 
                transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]),
            )
    else:
        test_dataset= datasets.Places365(
                "./data/",
                split='val',
                download=True,
                small=True, 
                transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]),
            )
    
    return test_dataset