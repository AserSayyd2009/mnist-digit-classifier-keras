# MNIST Digit Classification using Convolutional Neural Networks (CNN)

This repository contains a high-performance Convolutional Neural Network (CNN) implementation designed to recognize and classify handwritten digits ($0$ through $9$) using the benchmark **MNIST dataset**. The project is built leveraging the Keras API within TensorFlow to achieve near-perfect classification accuracy.

---

## 📌 Project Overview

Image classification tasks require spatial hierarchy preservation. Unlike standard Dense networks that flatten spatial data, this project implements a **CNN** to automatically and adaptively learn spatial hierarchies of features from the input images through convolutional and pooling layers.

---

## 📊 Dataset & Preprocessing

The model is trained on the **MNIST dataset**, which consists of grayscale images of handwritten digits.

- **Total Images:** 70,000
  - **Training Set:** 60,000 images
  - **Testing Set:** 10,000 images
- **Image Dimensions:** $28 \times 28$ pixels, single-channel (grayscale).

### Preprocessing Pipeline:
1. **Reshaping (Tensor Formatting):** The input dimensions are reshaped from $(28, 28)$ to $(28, 28, 1)$ to explicitly define the single color channel required by 2D Convolutional layers.
2. **Normalization:** Pixel values are converted from integers in the range $[0, 255]$ to floating-point numbers in the range $[0.0, 1.0]$ via standard division ($X / 255.0$) to stabilize gradient descent.
3. **Target Encoding:** The integer labels ($0$-$9$) are transformed into binary vectors using **One-Hot Encoding** via `to_categorical()`, converting a label like `5` into `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`.

---

## 🏗️ Model Architecture

The network consists of an optimized stack of gated feature extractors followed by a dense classifier. 

The structural blueprint and feature map transformations flow as follows:

$$\text{Input}(28 \times 28 \times 1) \longrightarrow \text{Conv2D}(32, 3\times3, \text{ReLU}) \longrightarrow \text{MaxPooling2D}(2\times2) \longrightarrow \text{Flatten} \longrightarrow \text{Dense}(64, \text{ReLU}) \longrightarrow \text{Dense}(10, \text{Softmax})$$

### Architectural Implementation (Keras):
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    # Feature Extraction Layer 1
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flattening spatial feature maps into a 1D vector
    Flatten(),
    
    # Fully Connected Classifier
    Dense(64, activation='relu'),
    Dense(10, activation='softmax') # 10 classes corresponding to digits 0-9
])