import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST
from torchvision import transforms

# Load dataset without any transforms
train_dataset_raw = PneumoniaMNIST(split='train', download=True, transform=None)

# Display some sample images before preprocessing
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    img, label = train_dataset_raw[i]
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')
plt.suptitle("Before Preprocessing")
plt.show()
transform = transforms.Compose([
    transforms.Resize((28, 28)),          # Resizes image to 28x28
    transforms.ToTensor(),                # Converts image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizes pixel values
])
# Load dataset with preprocessing
train_dataset_preprocessed = PneumoniaMNIST(split='train', download=True, transform=transform)

# Display sample images after preprocessing
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    img, label = train_dataset_preprocessed[i]
    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')
plt.suptitle("After Preprocessing")
plt.show()
