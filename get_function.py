#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */home/workspace/flowers/get_function.py
#
# PROGRAMMER: Anand Siva P V
# DATE CREATED: 08-03-2023
# REVISED DATE: 08-03-2023
# PURPOSE: Defines required helper functions for model training, evaluation and prediction 
# Define imports
# Imports python modules

import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
import json
import os


##-------------------------------------------------------------------------------------------------#
## 1.1 PARSE ARGUMENTS FOR TRAINING                                                                #
##-------------------------------------------------------------------------------------------------#

import argparse

def parse_arguments_train():
    """
    Parse command-line arguments using argparse.

    Returns:
        args: Parsed arguments
    """
    # create argument parser object
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')

    # add arguments to parser object
    parser.add_argument('data_dir', metavar='data_directory', help='the directory where the data is stored')
    parser.add_argument('--save_dir', metavar='save_directory', help='the directory where checkpoints will be saved')
    parser.add_argument('--arch', default='resnet18', choices=['efficientnet_v2_l','densenet121'], help='the architecture to use for the network')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='the learning rate to use for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=5120, help='the number of units in the hidden layer')
    parser.add_argument('--epochs', type=int, default=10, help='the number of epochs to train for')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')
   

    # parse arguments
    args = parser.parse_args()

    # return parsed arguments
    return args
    
##-------------------------------------------------------------------------------------------------#
## 1.2 DATA LOADER                                                                                 #
##-------------------------------------------------------------------------------------------------#

def get_data_loaders(path):
    '''
    This function takes the file paths for the dataset,
    loads the data, applies transforms, and returns data loaders for each data set.
    
    Inputs:
    - path: file path to the data directory

    Returns:
    - trainloader: PyTorch data loader for the training data set
    - validloader: PyTorch data loader for the validation data set
    - testloader: PyTorch data loader for the testing data set
    - train_dataset.class_to_idx: Dictionary maping flower class to dataloader index 
    '''
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30), # Randomly rotate images by up to 30 degrees
        transforms.RandomResizedCrop(224), # Randomly crop images to 224x224 pixels
        transforms.RandomHorizontalFlip(), # Randomly flip images horizontally
        transforms.ToTensor(), # Convert the image to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize the image using the specified means and standard deviations
    ])

    valid_transforms = test_transforms = transforms.Compose([
        transforms.Resize(255), # Resize the image to 255 pixels on the shorter side
        transforms.CenterCrop(224), # Crop the center 224x224 pixels of the image
        transforms.ToTensor(), # Convert the image to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize the image using the specified means and standard deviations
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms) # Load the training dataset with the training transforms
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms) # Load the validation dataset with the validation transforms
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms) # Load the testing dataset with the testing transforms

    # Define the dataloaders using the image datasets and the transforms
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) # Load the training dataset into batches of size 64, shuffle the data at each epoch
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64) # Load the validation dataset into batches of size 64
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64) # Load the testing dataset into batches of size 64

    return trainloader, validloader, testloader,train_dataset.class_to_idx
