#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */home/workspace/flowers/get_model.py
#
# PROGRAMMER: Anand Siva P V
# DATE CREATED: 03-03-2023
# REVISED DATE: 08-03-2023
# PURPOSE: Defines required functions for loading and training a pytorch model 
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
# 1. FUNCTIONS FOR test.py                                                                         #
##-------------------------------------------------------------------------------------------------#
##   1.1 IMPORT PRETRAINED MODEL                                                                   #
##-------------------------------------------------------------------------------------------------#

def import_model(model_name):
    """
    Imports a pretrained PyTorch model with the specified architecture.

    Args:
        model_name (str): Name of the model architecture to import.

    Returns:
        PyTorch model: The pretrained model with its weights frozen.
    """
    model = models.__dict__[model_name](pretrained=True);
    for param in model.parameters():
        param.requires_grad = False
    return model


##-------------------------------------------------------------------------------------------------#
##   1.2 LOAD AND CHANGE FINAL LAYERS OF MODEL                                                     #
##-------------------------------------------------------------------------------------------------#
# Resnet LOAD
def Load_model(model_name,hidden_units, learning_rate):

  """
    Loads the pre-trained model and modifies the final layers to create a custom classifier 
    for the flower classification task.

    Args:
    - model_name (str): the name of the pre-trained model to use from ['resnet18', 'efficientnet_v2_l', 'densenet121']
    - hidden_units (int): the number of neurons in the hidden layer of the custom classifier
    - learning_rate (float): the learning rate to use for training the model
    
    Returns:
    - model (torch model): the pre-trained model with custom classifier
    - criterion (torch criterion): the loss function used for training the model
    - optimizer (torch optimizer): the optimizer used for training the model
  """
# Load the pre-trained ResNet18 model
  model = import_model(model_name);
  # Set number of input features
  if model_name == 'efficientnet_v2_l':
    num_features = 1280
  elif model_name =='densenet121':
    num_features = 1024
  else:
    num_features = 512 
# Modify the final layers to create a custom classifier
  
  new_classifier = nn.Sequential(
                    nn.Linear(num_features, hidden_units), # add a linear layer with 512 output neurons
                    nn.ReLU(), # add a ReLU activation function
                    nn.Dropout(p=0.2), # add a dropout layer with probability of 0.2
                    nn.Linear(hidden_units, hidden_units), # add another linear layer with 5120 output neurons
                    nn.ReLU(), # add a ReLU activation function
                    nn.Dropout(p=0.2), # add a dropout layer with probability of 0.2
                    nn.Linear(hidden_units, 102), # add a linear layer with 102 output neurons (number of classes)
                    nn.LogSoftmax(dim=1) # add a log softmax activation function
                  )
  # Replacing the final layer of pretrained model with new_classifier
  if model_name in (['efficientnet_v2_l','densenet121']):
    model.classifier = new_classifier 
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0003)# use Adam optimizer with a learning rate of 0.0003
  else:
    model.fc = new_classifier 
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0003) 

  # Set the loss function for training the model
  criterion = nn.NLLLoss() # use negative log likelihood loss as the criterion
  return model, criterion, optimizer
