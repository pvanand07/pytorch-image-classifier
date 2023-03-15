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