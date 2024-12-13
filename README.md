# Pothole Detection Using Deep Learning

![Screenshot 2024-12-14 023346](https://github.com/user-attachments/assets/331120e8-2e7a-40a8-a4a8-546adad36d4b)


# Pothole Detection Using Deep Learning

This repository contains a PyTorch-based implementation for detecting potholes in images using deep learning. The project classifies images into two categories: "Pothole" and "Normal," and provides a Flask web application for users to upload images and receive predictions.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Data Augmentation](#data-augmentation)
- [Training Script](#training-script)
- [Evaluation](#evaluation)
  - [Confusion Matrix](#confusion-matrix)
  - [ROC Curve](#roc-curve)
- [Web Application](#web-application)
- [Results](#results)
- [Installation](#installation)
- [Conclusion](#conclusion)

## Introduction
Potholes pose a significant risk to road safety, and timely detection is crucial for maintaining infrastructure. This project uses deep learning to classify images into two categories: "Pothole" and "Normal," enabling automated pothole detection and reporting.

## Dataset
The dataset consists of images categorized as follows:
- **Pothole**: Images containing potholes.
- **Normal**: Images containing normal road conditions.

The dataset is split into:
- **Training**: 70%
- **Validation**: 15%
- **Testing**: 15%

## Requirements
Dependencies:
- Python 3.8+
- PyTorch
- Torchvision
- Flask
- Pillow
- scikit-learn
- Matplotlib
- Seaborn

Install dependencies using:
```bash
pip install -r requirements.txt

Model Architecture
The model is based on EfficientNet-B0, pretrained on ImageNet. The classifier is modified to output predictions for two classes:

Pothole
Normal
Model Layers:
EfficientNet-B0 backbone
Linear classifier layer with 2 output units (for binary classification)
Training
Data Augmentation
Training data is augmented with transformations including:

Resize to 224x224
Random Horizontal Flip
Random Rotation
Color Jitter (brightness, contrast, saturation, hue)
Normalization using ImageNet means and std
Training Script
Key features of the training script:

Early stopping with a patience of 5 epochs
Weighted cross-entropy loss to handle class imbalance
Adam optimizer with a learning rate scheduler
Run the training script with:

python train.py
Evaluation
Confusion Matrix
A confusion matrix is plotted to evaluate the model's performance across classes.

ROC Curve
ROC curves for each class are plotted to evaluate the model's discriminative ability, with the AUC (Area Under Curve) calculated.

Web Application
The Flask web application allows users to upload images and receive predictions of whether a pothole is present. The features include:

Predicted class: Pothole or Normal
Option to upload new images for testing
To run the Flask app:

python app.py
Access the web application at http://127.0.0.1:5000/.

Results
Test Accuracy: The model achieved high accuracy on the test set.
AUC-ROC: High AUC scores indicate reliable predictions.
Installation
To set up the project:

Clone this repository:
git clone <repository-url>
cd <repository-folder>
Install dependencies:
pip install -r requirements.txt
Conclusion
This project demonstrates the effectiveness of deep learning in detecting potholes from images. The model provides accurate predictions, aiding in the automated monitoring of road conditions.
