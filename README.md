# Pytorch Image Classification

## Project Summary

This project aims to classify flower images using a deep learning model trained with PyTorch. The model uses a pretrained feature extractor and is trained with data augmentation and normalization to improve accuracy. A command-line application is also developed to predict the class of a flower image and display the top K classes with associated probabilities. The application allows users to train a new network on a given dataset, predict the class, and display the top K classes with associated probabilities. The project demonstrates how to load data, train a deep learning model, and create a command-line interface to interact with the model.

## Files

### The following files are included in this repository:

`train.py`: This script is used to train a new network on a dataset of images. It allows users to set hyperparameters for learning rate, number of hidden units, and training epochs. The script also allows users to choose from at least two different architectures available from torchvision.models and to choose whether to train the model on a GPU. The training loss, validation loss, and validation accuracy are printed out as a network trains.

`predict.py`: This script is used to predict the class of a flower image and display the top K classes with associated probabilities. It allows users to load a trained model checkpoint, map class values to other category names using a JSON file, and to choose whether to use the GPU to calculate the predictions.

`get_function.py`: This script contains utility functions used in `train.py` and `predict.py`, including a function to load and preprocess images.

`get_model.py`: This script contains the function to load a pretrained feature extractor and define a new classifier.

## Usage

### Training

To train a new network, run the train.py script in the command line with the following arguments:
```
python train.py data_directory --arch "resnet18" --learning_rate 0.0003 --hidden_units 5120 --epochs 10 --gpu
```
- data_directory  :  help= 'the directory where the training data is stored'
- --save_dir      : type=str, the directory where checkpoints will be saved
- --arch          : default='resnet18', choices=['efficientnet_v2_l','densenet121'], help='the architecture to use for the network'
- --learning_rate : type=float, default=0.0003, help='the learning rate to use for the optimizer'
- --hidden_units  : type=int, default=5120, help='the number of units in the hidden layer'
- --epochs        : type=int, default=10, help= 'the number of epochs to train for'
- '--gpu'         : toggle to use GPU for training

### Prediction

To predict the class of a flower image, run the predict.py script in the command line with the following arguments:

```
python predict.py "/path/to/image" checkpoint --category_names cat_to_name.json --top_k 5 --gpu
```
- input_path : help= the path to the image file
- checkpoint : help= the path to the checkpoint file
- --top_k : type=int, default=5, help= return the top K most likely classes
- --category_names : help= the file containing the category names
- --gpu : help = toggle to use GPU for inference

## Dependencies

### This project requires the following dependencies:

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib

These dependencies can be installed using pip and the requirements.txt file included in this repository:

```
pip install -r requirements.txt
```

## Acknowledgements

This project was completed as part of the Udacity AI and ML Nanodegree program. The flower dataset used in this project is the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) by Maria-Elena Nilsback and Andrew Zisserman

