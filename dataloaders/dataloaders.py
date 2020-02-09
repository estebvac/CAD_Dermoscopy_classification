import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def data_loader(path_train, train_transforms, path_test=None, test_transforms=None,
                valid_size=None, batch_size=32):
    if (path_test == None and valid_size == None):
        train_data = torchvision.datasets.ImageFolder(root=path_train, transform=train_transforms)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        dataloaders = {'train': train_loader}
        data_sets = {'train': train_data}

    else:
        train_data = torchvision.datasets.ImageFolder(root=path_train, transform=train_transforms)
        valid_data = torchvision.datasets.ImageFolder(root=path_train, transform=test_transforms)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        # Define a random seed
        np.random.seed(42)
        np.random.shuffle(indices)

        # Split the index
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler)

        if (path_test == None and valid_size != None):
            dataloaders = {'train': train_loader, 'val': valid_loader}
            data_sets = {'train': train_data, 'val': valid_data}
        elif (path_test != None and valid_size != None):
            test_data = torchvision.datasets.ImageFolder(root=path_test, transform=test_transforms)
            test_loader = DataLoader(test_data, batch_size=batch_size)
            dataloaders = {'train': train_loader, 'val': valid_loader, 'test': test_loader}
            data_sets = {'train': train_data, 'val': valid_data, 'test': test_data}

    return dataloaders, data_sets
