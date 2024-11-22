import os
from collections import Counter
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================
# Step 1: Configuration and Hyperparameters
# =====================================================
# Dataset path
data_path = "/CSCI2952X/datasets/affectnet_3750subset"

# Hyperparameters
batch_size = 512
num_classes = 8
num_epochs = 200
learning_rate = 0.03

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# =====================================================
# Step 2: Data Preparation
# =====================================================
# Define transformations for training and validation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation: Random horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Augmentation for color
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Load datasets using ImageFolder
train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=train_transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=test_transform)

# Print dataset distributions
print("Training Dataset Distribution:")
print(Counter(train_dataset.targets))

print("Test Dataset Distribution:")
print(Counter(test_dataset.targets))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# =====================================================
# Step 3: Model Definition
# =====================================================
# Load pretrained ResNet18 and modify the final layer
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =====================================================
# Step 4: Training Loop
# =====================================================
print("\nStarting training...")
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print training loss for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    # =================================================
    # Step 5: Validation Loop
    # =================================================
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

# =====================================================
# Step 6: Cleanup
# =====================================================
torch.cuda.empty_cache()  # Clear CUDA cache
print("Training complete.")