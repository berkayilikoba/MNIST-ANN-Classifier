# MNIST-ANN-Classifier
Digit recognition on MNIST using a fully connected neural network implemented with Keras.

This project implements a simple Artificial Neural Network (ANN) to classify handwritten digits from the MNIST dataset using Keras and TensorFlow.

## Features

- Fully connected neural network with two hidden layers
- Uses ReLU activation in hidden layers and softmax in output layer
- Early stopping and model checkpointing during training
- Accuracy and loss visualization for training and validation sets
- Confusion matrix and classification report for test evaluation

## Dataset

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. It is split into 60,000 training images and 10,000 test images.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

# Usage

Run the scripts in the following order:

1. **model.py** — Defines the ANN model architecture.  
2. **train.py** — Loads data, trains the model, and saves the best model.  
3. **predict.py** — Loads the saved model, performs predictions on test data, and visualizes results.






