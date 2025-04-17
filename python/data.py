#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:16:57 2019

@author: server
"""

###################
### DATALOADING ###
###################

### Imports
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

### Defining data preprocessing step

# training set
train_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

# test and valid sets
check_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

# Define the root directory containing 'benign' and 'malign' folders
root_dir = '../data/'

# Load the entire dataset
full_dataset = ImageFolder(root_dir, transform=train_transforms)

# Get the indices and labels
indices = list(range(len(full_dataset)))
labels = [full_dataset.targets[i] for i in indices]

# Split into train, temp (test + valid)
train_indices, temp_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

# Split temp into test and valid
test_indices, valid_indices = train_test_split(temp_indices, test_size=0.5, stratify=[labels[i] for i in temp_indices], random_state=42)

# Create subsets
trainset = Subset(full_dataset, train_indices)
testset = Subset(full_dataset, test_indices)
validset = Subset(full_dataset, valid_indices)

# Apply appropriate transforms for each subset
trainset.dataset.transform = train_transforms
testset.dataset.transform = check_transforms
validset.dataset.transform = check_transforms

# Create DataLoaders
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
testloader = DataLoader(testset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)
validloader = DataLoader(validset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

loaders = {'train':trainloader, 'test':testloader, 'valid': validloader}

### Sanity check
# print(len(trainset), len(testset), len(validset))
# print(len(trainloader), len(testloader), len(validloader))
