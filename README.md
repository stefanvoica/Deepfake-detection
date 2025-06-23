# Deepfake Detection using CNN 

This repository contains the code and report for a deepfake image classification project, developed for the University of Bucharest competition hosted on Kaggle. The main goal is to build a reliable and scalable deep learning model capable of distinguishing between real and AI-generated (deepfake) face images.

---

## 🎯 Objective

- Build an accurate image classification model using Convolutional Neural Networks (CNN)
- Apply modern training techniques: regularization, label smoothing, learning rate scheduling
- Combine multiple models using ensembling for increased robustness

---
## 🧠 CNN Model Architecture

The Convolutional Neural Network consists of:
- **4 convolutional blocks**: Conv2D → BatchNorm → ReLU
- **2 MaxPooling layers**: spatial downsampling
- **Dropout layers**: for regularization and reduced overfitting
- **2 fully connected layers**: with ReLU activations and final classification output

---

## ⚙️ Training Details

- **Loss function**: `CrossEntropyLoss` with **label smoothing** (`0.1`) to improve generalization
- **Optimizer**: `AdamW` with weight decay (`1e-3`)
- **Learning Rate Scheduler**: `CosineAnnealingLR`  
  ↓ gradually reduces learning rate from `1e-3` to `1e-5` across 160 epochs
- **Metrics**: Accuracy and average loss on both training and validation sets
- **Augmentations**:
  - Random rotations
  - Horizontal flips

---

## 🧪 Model Ensemble Strategy

To boost performance and stability, the final classification relied on **model ensembling**:
- Multiple CNNs were trained independently with different augmentations and random seeds
- The best-performing models (~93% accuracy) were saved
- Their predictions were averaged (soft-voting) during inference

This approach mitigated variance and improved final validation accuracy.
![image](https://github.com/user-attachments/assets/ce091c26-d0df-4217-a0c4-ae5e0a066320)

---

## 💡 Technologies Used

- **PyTorch**
- **OpenCV**  
- **NumPy, Pandas**
- **Matplotlib**

---

## 📌 Author

**Ștefan-Alexandru Voica**  
Faculty of Mathematics and Computer Science  
University of Bucharest – 2025  
📄 See full details in `report.pdf`
📧 stefan.voica04@gmail.com  
