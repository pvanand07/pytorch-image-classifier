#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */home/workspace/flowers/train.py
#
# PROGRAMMER: Anand Siva P V
# DATE CREATED: 08-03-2023
# REVISED DATE: 08-03-2023
# PURPOSE: Trains a pytorch model based on transfer learning on a given dataset
# Define imports

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
import os
from get_function import parse_arguments_train

# Get input arguments
arg = parse_arguments_train()
print(arg)

  # Checking gpu availability and assigning device
if arg.gpu and torch.cuda.is_available():
    device = "cuda"
    print("Using gpu for computing")
elif arg.gpu and not (torch.cuda.is_available()):
    device = "cpu"
    print("gpu unavailable, Using cpu for computing")
else:
    device = "cpu"
    print("Using cpu for computing")

# Presprocessing image data for training, validation and testing
trainloader, validloader, testloader, class_to_idx = get_data_loaders(arg.data_dir)

# Obtaining model, criterion and optimizer
model, criterion, optimizer = Load_model(arg.arch, arg.hidden_units, arg.learning_rate)

# Training model
model = train_model(trainloader, validloader, model, criterion, optimizer, device, arg.epochs)
# Creating save directory if it doesn't exists
if (arg.save_dir is not None) and (not os.path.exists(arg.save_dir)):
    os.mkdir(arg.save_dir)