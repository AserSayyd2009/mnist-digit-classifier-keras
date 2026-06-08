# MNIST Digit Classification using CNN (Keras / TensorFlow)

A high-performance **Convolutional Neural Network (CNN)** built with Keras to classify handwritten digits (0–9) using the **MNIST dataset**, achieving ~99% accuracy.

---

## 🚀 Project Goal

This project demonstrates how CNNs outperform traditional dense neural networks in image classification by learning spatial hierarchies of features directly from raw pixel data.

---

## 📊 Dataset

- **Total samples:** 70,000 images  
- **Training set:** 60,000 images  
- **Test set:** 10,000 images  
- **Image size:** 28 × 28 pixels  
- **Channels:** 1 (grayscale)  
- **Classes:** 10 digits (0–9)

---

## 🧹 Data Preprocessing

- Reshaping images from `(28, 28)` → `(28, 28, 1)`
- Normalizing pixel values from `[0, 255]` → `[0, 1]`
- One-hot encoding labels using `to_categorical()`

---

## 🧠 Model Architecture

```
Input (28×28×1)
      ↓
Conv2D (32 filters, 3×3, ReLU, stride=2)
      ↓
MaxPooling2D (2×2)
      ↓
Conv2D (64 filters, 2×2, ReLU)
      ↓
MaxPooling2D (2×2)
      ↓
Flatten
      ↓
Dense (128, ReLU)
      ↓
Dense (10, Softmax)
```

---

## 🏗️ Model Implementation

```python
model = Sequential([
    Conv2D(32, (3,3), strides=(2,2), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (2,2), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

## ⚙️ Training Setup

- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Metrics: Accuracy  
- Epochs: 11  
- Validation Split: 0.2  

### Callbacks:
- EarlyStopping (prevents overfitting)
- ModelCheckpoint (saves best model)

```python
EarlyStopping(patience=3, restore_best_weights=True)

ModelCheckpoint(
    "best_model.keras",
    monitor='val_accuracy',
    save_best_only=True
)
```

---

## 📈 Evaluation

- Classification Report (precision / recall / F1-score)
- Confusion Matrix visualization
- Test accuracy: ~99%

```python
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print(classification_report(y_test_labels, y_pred_labels))
```

---

## 🔍 Inference Example

```python
sample = X_test[i].reshape(1, 28, 28, 1)
pred = model.predict(sample)

print("Predicted:", np.argmax(pred))
print("True:", np.argmax(y_test[i]))
```

---

## 📦 Requirements

```bash
tensorflow
keras
numpy
pandas
scikit-learn
matplotlib
```

---

## 📁 Project Structure

```
MNIST-CNN/
│
├── CNN_classification.ipynb
├── best_model.keras
├── README.md
└── requirements.txt
```

---

## 🚀 Key Highlights

- Lightweight CNN architecture
- Fast convergence with Adam optimizer
- EarlyStopping + ModelCheckpoint
- ~99% accuracy on MNIST
- Full evaluation pipeline included

---

## 🔮 Future Improvements

- Add BatchNormalization
- Add Dropout layers
- Data augmentation (rotation, shift)
- Deeper CNN architecture
- Compare with MLP baseline

---

## 🧠 Summary

This project implements a complete deep learning pipeline:
**Data → Preprocessing → CNN → Training → Evaluation → Prediction**
