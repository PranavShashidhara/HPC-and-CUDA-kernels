# gpu_mnist_edge_demo.py

import torch
import torch.nn as nn
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt

# Download MNIST and save one digit as PNG
dataset = MNIST(root=".", download=True, train=True)
img_pil, label = dataset[0]  
img_path = "sample_digit.png"
img_pil.save(img_path)
print(f"Saved MNIST digit as {img_path} (label={label})")

# Load the saved image (grayscale)
img = Image.open(img_path).convert("L")
transform = ToTensor()
img_tensor = transform(img).unsqueeze(0)  # add batch dim

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_tensor = img_tensor.to(device)

# Define a simple edge-detection convolution
conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(device)

# Move model to device
conv.weight.data = torch.tensor([[[[-1., -1., -1.],
                                   [-1.,  8., -1.],
                                   [-1., -1., -1.]]]], device=device)

# Apply convolution
output = conv(img_tensor)

# Move back to CPU for saving
output_img = output.squeeze().detach().cpu()

# Save original and processed images side by side
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Edge Detection (GPU)")
plt.imshow(output_img, cmap="gray")
plt.axis("off")

plt.savefig("output_edge_digit.png")
print("Edge-detected image saved as output_edge_digit.png")
