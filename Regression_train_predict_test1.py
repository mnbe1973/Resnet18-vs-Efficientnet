# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:11:29 2023

@author: Magopti
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx])
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# Specify your image files and labels
image_files = ['obj1__0.png', 'obj1__5.png', 
               'obj1__10.png', 'obj1__15.png', 
               'obj1__20.png', 'obj1__25.png']
labels = [0, 5, 10, 15, 20, 25]  # your regression values

# Define the transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the CustomDataset
dataset = CustomDataset(image_files, labels, transform)

# Split dataset into training set and validation set
train_size = int(1.0 * len(dataset))  # 80% for training
val_size = int(1.0 * len(dataset)) #len(dataset) - train_size  # 20% for validation
#train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloader
train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)

# Initialize ResNet
model = models.resnet18(pretrained=True)

# Freeze the pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = model.fc.in_features
model.fc = nn.Linear(fc_inputs, 1)  # For regression

# Convert model to be used on GPU
model = model.to(device)

# Define Optimizer and Loss Function
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training
for epoch in range(1000):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)  # Cast to float for regression
        
        # Forward pass
        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)  # Cast to float for regression
            outputs = model(images)
            loss = loss_func(outputs, labels)
            total_loss += loss.item()
        print('Validation loss:', total_loss / len(val_loader))
save_path = 'resnet18_trained.pth'
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")