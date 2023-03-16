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

##-------------------------------------------------------------------------------------------------#
##   1.3 TRAIN MODEL                                                                               #
##-------------------------------------------------------------------------------------------------#

def train_model(trainloader, validloader, model, criterion, optimizer, device, epochs):
    """
    Trains a PyTorch model on a given dataset using the specified hyperparameters.

    Args:
        trainloader (torch.utils.data.DataLoader): PyTorch DataLoader for the training set
        validloader (torch.utils.data.DataLoader): PyTorch DataLoader for the validation set
        model (torch.nn.Module): PyTorch model to be trained
        criterion (torch.nn.modules.loss._Loss): loss function used for training the model
        optimizer (torch.optim.Optimizer): optimization algorithm used for updating the model's parameters
        device (str): the device (CPU or GPU) on which the model should be trained
        epochs (int): the number of epochs for which to train the model

    Returns:
        model (torch.nn.Module): trained PyTorch model 
    """
    # Move the model to the preferred device
    model.to(device)

    # Initialize variables
    steps = 0
    running_loss = 0
    print_every = 5
    train_losses, test_losses = [], []

    # Loop over each epoch, using tqdm to print out progress during each iteration
    for epoch in tqdm(range(epochs)):
      
        # Loop over the training dataset in batches
        for inputs, labels in tqdm(trainloader):
            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass, compute loss, backpropagate, and update parameters
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad() # Set the gradients to zero to clear the gradients from previous step 
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Every print_every steps, calculate validation loss and accuracy and print metrics
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader: # Iterate through each batch of inputs and labels
                        inputs, labels = inputs.to(device), labels.to(device) # Push inputs and labels to available device
                        logps = model.forward(inputs) # calculating log-probabilities using model.forward()
                        batch_loss = criterion(logps, labels) # Calculating batch loss using criterion

                        test_loss += batch_loss.item() # Add each batch loss to obtain total test loss 

                        # Calculate accuracy
                        ps = torch.exp(logps) # converting log probabilties to probabilites 
                        top_p, top_class = ps.topk(1, dim=1) # Obtaining top class and probabilities
                        equals = top_class == labels.view(*top_class.shape) # Comparing obtained top classes and their labels
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # Calculating accuracy by taking the mean of equals tensor

                # Print metrics
                print(f"Epoch {round(steps/len(trainloader),2)}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}")

                # At completion of epoch, append training and validation losses to lists and reset running loss
                train_losses.append(running_loss)
                test_losses.append(test_loss)
                running_loss = 0
                model.train()

    # Return the lists of training and validation losses
    return model

##-------------------------------------------------------------------------------------------------#
# 2. FUNCTIONS FOR predict.py                                                                      #
##-------------------------------------------------------------------------------------------------#
##   2.1 LOAD CHECKPOINT AND REBUILD MODEL                                                         #
##-------------------------------------------------------------------------------------------------#
def load_checkpoint(file_path,arch,hidden_units, learning_rate):
    """
    Load a pretrained model checkpoint and rebuild the model using the saved state dictionary.
    
    Args:
    file_path (str): Path to the saved checkpoint file.
    
    Returns:
    model (torch.nn.Module): The reconstructed model.
    """
    
    # Load the pretrained model
    model,_,_ = Load_model(arch,hidden_units, learning_rate); 
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Load the state dict with torch.load
    checkpoint = torch.load(file_path, map_location=torch.device(device));
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']
    
    # Load the state dict into the model
    model.load_state_dict(state_dict);
    
    return model,class_to_idx
