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

