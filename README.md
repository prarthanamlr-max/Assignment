# 📘 Deep Learning Project: CNN, RNN, and GAN

## 📌 Overview
This project demonstrates the implementation of three fundamental deep learning models:

- Convolutional Neural Networks (CNN) → Image Classification  
- Recurrent Neural Networks (RNN, LSTM, GRU) → Sentiment Analysis  
- Generative Adversarial Networks (GAN) → Image Generation  

The models are trained and evaluated on standard datasets to understand their performance and behavior.

---

## 📂 Datasets Used

### 🖼️ CIFAR-10
- 60,000 images (32×32)
- 10 classes
- Used for CNN image classification

### 📝 IMDB Dataset
- 50,000 movie reviews
- Binary sentiment classification
- Used for RNN models

### 👕 Fashion-MNIST
- 70,000 grayscale images (28×28)
- 10 classes
- Used for GAN image generation

---

## 🧠 Models Implemented

### 🔹 CNN (Image Classification)
- Custom CNN architecture
- Transfer Learning using MobileNetV2
- Achieved accuracy up to **88.51%**

### 🔹 RNN Models (Text Classification)
- Simple RNN
- LSTM
- GRU

**Performance:**
- RNN: ~30%
- LSTM: ~40%
- GRU: ~40%

### 🔹 GAN (Image Generation)
- Generator + Discriminator architecture
- Trained on Fashion-MNIST
- Generated images improved over epochs

---

## ⚙️ Technologies Used
- Python
- PyTorch
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab

---

## 🚀 Training Details

| Parameter       | Value                          |
|----------------|--------------------------------|
| Batch Size     | 32–128                         |
| Epochs         | 3–50                           |
| Learning Rate  | 0.001 (CNN/RNN), 0.0002 (GAN) |
| Optimizer      | Adam                           |

---

## 📊 Results Summary

| Model         | Accuracy |
|--------------|----------|
| CNN          | 88.51%   |
| MobileNetV2  | 59.32%   |
| RNN          | ~30%     |
| LSTM         | ~40%     |
| GRU          | ~40%     |

---

## 📈 Key Observations
- CNN performed better than transfer learning for Fashion-MNIST
- LSTM and GRU outperformed basic RNN
- GAN showed gradual improvement in image quality
- Mode collapse observed in GAN training

---

## ⚠️ Limitations
- Limited training epochs
- GAN instability
- Basic architectures used

---

## 🔮 Future Work
- Improve model architectures
- Use advanced GAN variants (WGAN, cGAN)
- Apply transformer-based models
- Perform extensive hyperparameter tuning

---

## 🔗 GitHub Repository
https://github.com/prarthanamlr-max/Assignment.git

---

## 📜 License
This project is for academic purposes only.

---

## 👩‍💻 Author
**Prarthana M Rao**
