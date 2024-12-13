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

