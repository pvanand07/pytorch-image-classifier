#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */home/workspace/flowers/predict.py
#
# PROGRAMMER: Anand Siva P V
# DATE CREATED: 03-03-2023
# REVISED DATE: 08-03-2023
# PURPOSE: Predicts the class of an image using a pretrained model
# Define imports
# Runs the following code only if the file is run as a script
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
import os
from get_function import parse_arguments_predict
from get_function import get_data_loaders
import warnings

# Hide all warnings
warnings.filterwarnings("ignore")

# Get input arguments
arg = parse_arguments_predict()
print(arg)

# Determine device to be used for computation
if arg.gpu and torch.cuda.is_available():
    device = "cuda"
    print("Using gpu for computing")
elif arg.gpu and not (torch.cuda.is_available()):
    device = "cpu"
    print("gpu unavailable, Using cpu for computing")
else:
    device = "cpu"
    print("Using cpu for computing")

# Load model from checkpoint
model, class_to_idx = load_checkpoint(arg.checkpoint, arch="resnet18", 
                                    hidden_units=5120, learning_rate=0.0003
                                     )

# Preprocess image
img_t = process_image(arg.input_path)

# Get idx_to_class dictionary
idx_to_class = {v: k for k, v in class_to_idx.items()}