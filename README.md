# 🖼️ CIFAR-10 Image Classification Using CNN  

This project implements **image classification** on the **CIFAR-10 dataset** using **Convolutional Neural Networks (CNNs)** in **Google Colab**. The model is trained to recognize **10 different categories** of images with high accuracy.  

---

## 🔍 **Project Overview**  
The **CIFAR-10 dataset** is a well-known benchmark dataset for machine learning and computer vision. It contains **60,000 images** categorized into **10 different classes**:  
✈️ Airplane, 🚗 Automobile, 🐦 Bird, 🐱 Cat, 🦌 Deer, 🐶 Dog, 🐸 Frog, 🐴 Horse, 🛳️ Ship, and 🚚 Truck.  

### 🎯 **Objectives of the Project**  
✔️ Implement a **deep learning model** using CNNs.  
✔️ Achieve **high classification accuracy** on the CIFAR-10 dataset.  
✔️ Perform **data preprocessing and augmentation** for better training.  
✔️ Use **Google Colab GPU** to speed up training.  
✔️ Visualize **model performance** using accuracy and loss graphs.  

---

## 📊 **Dataset Details**  
- **Size**: 60,000 images (50,000 for training, 10,000 for testing)  
- **Image Size**: 32×32 pixels  
- **Classes**: 10  

---

## 🛠️ **Technologies & Tools Used**  
✅ **Programming Language**: Python  
✅ **Deep Learning Framework**: TensorFlow / Keras  
✅ **Libraries Used**: NumPy, Matplotlib, OpenCV, Seaborn  
✅ **Training Environment**: Google Colab (with GPU)  

---

## 🚀 **Model Architecture**  
The CNN architecture consists of:  
✔️ **Convolutional Layers** for feature extraction  
✔️ **Batch Normalization** for stable learning  
✔️ **Max Pooling Layers** to reduce dimensionality  
✔️ **Dropout Layers** to prevent overfitting  
✔️ **Fully Connected Layers** for classification  

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```
## 📈 Model Training & Evaluation  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Batch Size**: 64  
- **Epochs**: 50  
- **Training Accuracy**: ~85%  
- **Testing Accuracy**: ~80%  

### 📊 Results & Performance  
✅ **Training & Validation Accuracy Graph**  
✅ **Confusion Matrix for Predictions**  
✅ **Classification Report with Precision, Recall, F1-score**  

---

## 🎯 How to Run the Notebook  
1. **Open Google Colab**.  
2. **Upload `cifar10_classification.ipynb`**.  
3. **Run all cells** (Training might take 5-10 minutes).  
4. **View model performance graphs and test results**.  

---

## 📜 Future Improvements  
🚀 **Use Transfer Learning** (Pretrained CNNs like ResNet, VGG-16).  
🚀 **Hyperparameter Tuning** for better accuracy.  
🚀 **Deploy the model** as a web application using Flask or Streamlit.  

---

## 🔗 Project Links  
🔗 **GitHub Repository**: [CIFAR-10-Classifier](https://github.com/sarath2230/CIFAR-10-Classifier)  
