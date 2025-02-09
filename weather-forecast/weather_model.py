import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Sample Weather Dataset (Temperature Prediction)
data = {
    "Temperature": [30, 32, 28, 35, 31, 29, 27, 33, 36, 34, 28, 30, 25, 22, 31],
    "Humidity": [70, 65, 80, 50, 75, 85, 90, 55, 40, 45, 78, 68, 92, 95, 63],
    "WindSpeed": [10, 12, 8, 15, 9, 7, 6, 14, 18, 13, 10, 11, 5, 4, 12],
    "Pressure": [1010, 1008, 1012, 1005, 1007, 1013, 1015, 1006, 1004, 1009, 1011, 1010, 1014, 1016, 1008]
}

df = pd.DataFrame(data)

# Split features (X) and target (Y)
X = df[["Humidity", "WindSpeed", "Pressure"]].values
Y = df["Temperature"].values.reshape(-1, 1)

# Normalize the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
Y_scaled = scaler_y.fit_transform(Y)

# Convert to PyTorch tensors and move to GPU
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32).to(device)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)


# Define a 10-Layer Neural Network Model
class WeatherPredictor(nn.Module):
    def __init__(self):
        super(WeatherPredictor, self).__init__()
        self.fc1 = nn.Linear(3, 128)   # Input: 3 -> 128 neurons
        self.fc2 = nn.Linear(128, 256)  # 128 -> 256 neurons
        self.fc3 = nn.Linear(256, 512)  # 256 -> 512 neurons
        self.fc4 = nn.Linear(512, 512)  # 512 -> 512 neurons
        self.fc5 = nn.Linear(512, 256)  # 512 -> 256 neurons
        self.fc6 = nn.Linear(256, 128)  # 256 -> 128 neurons
        self.fc7 = nn.Linear(128, 64)   # 128 -> 64 neurons
        self.fc8 = nn.Linear(64, 32)    # 64 -> 32 neurons
        self.fc9 = nn.Linear(32, 16)    # 32 -> 16 neurons
        self.fc10 = nn.Linear(16, 1)    # Output: 16 -> 1 (Temperature)
        
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.fc10(x)  # No activation on last layer (Regression problem)
        return x


# Initialize model and move to GPU
model = WeatherPredictor().to(device)

# Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the Model
epochs = 500
for epoch in range(epochs):
    model.train()

    predictions = model(X_train)
    loss = criterion(predictions, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Training complete.")

# Evaluate the Model
model.eval()
with torch.no_grad():
    Y_pred = model(X_test)
    Y_pred = scaler_y.inverse_transform(Y_pred.cpu().numpy())  
    Y_actual = scaler_y.inverse_transform(Y_test.cpu().numpy())

# Plot Actual vs Predicted Values
plt.scatter(Y_actual, Y_pred, color='blue')
plt.plot([min(Y_actual), max(Y_actual)], [min(Y_actual), max(Y_actual)], linestyle='--', color='red')
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.show()