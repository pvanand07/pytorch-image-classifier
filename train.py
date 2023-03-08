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