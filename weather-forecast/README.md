# Weather Forecast Model & GPU Test Model

This repository contains a **Weather Forecast Model** using **PyTorch** and a **GPU Test Model** to verify CUDA support.

## 🚀 Setup & Installation

### **1️⃣ Create a Conda Environment**
Run the following command to create and activate the environment:
```bash
conda create --name mlgpu python=3.10 -y
conda activate mlgpu
```

### **2️⃣ Install PyTorch with CUDA Support**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### **3️⃣ Verify GPU Availability**
To test if PyTorch detects your NVIDIA GPU, run:
```bash
python gpu_test.py
```

If your GPU is properly configured, the output should be similar to:
```
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 4070
```

## 📌 **Weather Forecast Model**
### **🔹 Model Overview**
This model predicts temperature based on **Humidity, Wind Speed, and Pressure**.
- **Input Features:** `Humidity`, `Wind Speed`, `Pressure`
- **Architecture:** Deep Neural Network (10 layers)
- **Framework:** PyTorch

### **🛠 Training the Model**
To train the model, run:
```bash
python weather_model.py
```
Example output:
```
Using device: cuda
Epoch [50/500], Loss: 0.1108
Epoch [100/500], Loss: 0.0352
...
Epoch [500/500], Loss: 0.0567
Training complete.
```

### **📊 Evaluate Predictions**
After training, the model plots **Actual vs. Predicted Temperature**:
```bash
python evaluate.py
```
This will generate a graph showing the performance of the model.

## 📌 **GPU Test Model**
### **🔹 Verify GPU Acceleration**
To confirm that your **GPU is being used for ML workloads**, run:
```python
torch.cuda.is_available()
```
Expected output:
```
True
```

### **🔹 Check Available GPU**
Run:
```python
import torch
print(torch.cuda.get_device_name(0))
```
Expected output:
```
NVIDIA GeForce RTX 4070
```

## 📜 **File Structure**
```
📂 weather_forecast_gpu
 ├── gpu_test.py          # Script to verify GPU availability
 ├── weather_model.py     # Neural network for weather prediction
 ├── evaluate.py          # Plot actual vs. predicted temperatures
 ├── README.md            # This documentation file
```

## ⚡ **Troubleshooting**
- **PyTorch doesn't detect GPU?**
  - Ensure CUDA is installed: `nvcc --version`
  - Try installing a compatible PyTorch version.

- **Training is slow?**
  - Confirm the model is using GPU: `print(torch.cuda.is_available())`

## 📢 **Contribute**
Feel free to open an **issue** or **pull request** if you want to improve this repository!

---
📌 **Author:** [Your Name]  
📌 **License:** MIT  
📌 **GitHub:** [Your Repo Link]

