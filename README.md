# CNN Digit Classifier (MNIST)

This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch to classify handwritten digits (0–9).

## Project Goal
Build and train a CNN model to recognize handwritten digits using the MNIST dataset.

## Dataset
MNIST dataset
- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 grayscale

## Model Architecture

CNN Architecture:

Conv Layer 1  
Input: 1×28×28  
Output: 32×26×26  

Conv Layer 2  
Input: 32×26×26  
Output: 64×24×24  

Max Pooling  
Output: 64×12×12  

Fully Connected Layer  
9216 → 128  

Output Layer  
128 → 10 classes

## Training

Loss Function:
CrossEntropyLoss

Optimizer:
Adam

Epochs:
5

Example Training Output:

Epoch 1 Loss: 128  
Epoch 2 Loss: 36  
Epoch 3 Loss: 22  
Epoch 4 Loss: 15  
Epoch 5 Loss: 12  

## Files

model.py  
Defines the CNN architecture.

train.py  
Trains the CNN model using MNIST dataset.

predict.py  
Loads trained model and predicts digits from images.

digit_cnn.pth  
Saved trained model weights.

## Example Prediction

Input image: handwritten digit  
Output: predicted digit (0–9)

## Technologies Used

Python  
PyTorch  
Torchvision  
PIL

## Learning Outcomes

Understanding CNN architecture  
Training deep learning models  
Saving and loading models  
Building inference pipelines