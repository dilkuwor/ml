import matplotlib.pyplot as plt

# Define layers (number of neurons per layer)
layers = [3, 128, 256, 512, 512, 256, 128, 64, 32, 16, 1]
layer_positions = range(len(layers))

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(-1, len(layers))
ax.set_ylim(-1, max(layers) * 0.005)  # Adjust spacing

ax.set_title("Neural Network Architecture - 10 Layers")

# Plot each layer
for i, neurons in enumerate(layers):
    for j in range(min(neurons, 10)):  # Limit to 10 neurons per layer for readability
        ax.scatter(i, j * 0.1, s=100, color="blue")  # Nodes

# Draw connections between layers
for i in range(len(layers) - 1):
    for j in range(min(layers[i], 10)):
        for k in range(min(layers[i + 1], 10)):
            ax.plot([i, i + 1], [j * 0.1, k * 0.1], color="gray", alpha=0.5)

# Add layer labels (Fixed issue)
for i in range(len(layers)):
    ax.text(i, 1.2, f"Layer {i+1}", fontsize=10, ha='center', color='red')

plt.axis("off")  # Hide axes
plt.show()