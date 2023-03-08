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
