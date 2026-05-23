# MNIST Digit Classifier using Keras

A beginner-friendly Deep Learning project built with Keras and TensorFlow to classify handwritten digits from the MNIST dataset.

## Overview

This project trains a simple Artificial Neural Network (ANN) on the MNIST handwritten digits dataset.  
The model is capable of recognizing digits from 0 to 9 with high accuracy.

The project includes:
- Data preprocessing
- Model training
- Model evaluation
- Saving and loading the trained model
- Predicting custom samples
- Visualization of predictions

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## Dataset

This project uses the MNIST dataset provided directly by Keras.

MNIST contains:
- 60,000 training images
- 10,000 testing images
- Grayscale handwritten digits (28x28 pixels)

---

## Model Architecture

The neural network consists of:
1. Input layer
2. Dense hidden layer with ReLU activation
3. Output layer with Softmax activation

---

## Training

The model was trained using:
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metric: Accuracy

---

## Results

The model achieves high accuracy on the MNIST test dataset.

---

## Example Prediction

The notebook includes examples of:
- Predicting handwritten digits
- Displaying test images
- Comparing predicted and actual labels

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt