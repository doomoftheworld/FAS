from common_imports import *

def get_tiny_imagenet_dataset(train_path, test_path):
    # Define the needed transformations
    required_transform = [ 
                            v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),  
                            v2.ToTensor(),
                        #   v2.Resize(size=(256, 256)), 
                        #   v2.RandomCrop(256, padding=12),
                        #   v2.RandomRotation(10),
                        #   v2.RandomInvert(),
                        #   v2.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                        #   v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                        #   v2.RandomErasing(),
                        #   v2.RandomHorizontalFlip(),   
                        ]

    train_dataset= datasets.ImageFolder(
            root=train_path,
            transform=v2.Compose(required_transform),
        )

    test_dataset= datasets.ImageFolder(
            root=test_path,
            transform=v2.Compose([v2.ToTensor()]),
            # transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(224, 224))]),
        )
    
    return train_dataset, test_dataset

def get_tiny_imagenet_without_transform(train_path, test_path):

    train_dataset= datasets.ImageFolder(
            root=train_path,
            transform=v2.Compose([v2.ToTensor()]),
        )

    test_dataset= datasets.ImageFolder(
            root=test_path,
            transform=v2.Compose([v2.ToTensor()]),
        )
    
    return train_dataset, test_dataset