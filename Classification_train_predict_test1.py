import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image
# Constants for the classification task
NUM_OBJECTS = 4  # Example value, change based on your data
NUM_COLORS = 4   # Example value, change based on your data

class CustomDataset(Dataset):
    def __init__(self, image_files, object_labels, color_labels, transform=None):
        self.image_files = image_files
        self.object_labels = object_labels
        self.color_labels = color_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx])
        if self.transform:
            img = self.transform(img)
        obj_label = self.object_labels[idx]
        color_label = self.color_labels[idx]
        return img, (obj_label, color_label)

# Example: for 2 objects and 2 colors
# 'obj1_red.png', 'obj1_blue.png', 'obj2_red.png', 'obj2_blue.png'
# Object labels: 0 for obj1, 1 for obj2
# Color labels: 0 for red, 1 for blue

image_files = ['ob1_yellow.png', 'ob2_green.png', 'ob3_red.png', 'ob4_blue.png']
object_labels = [0, 1, 2, 3]
color_labels = [0, 1, 2, 3]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomDataset(image_files, object_labels, color_labels, transform)

train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)

model = models.resnet18(pretrained=True)

# Modify the ResNet model for multi-task learning
for param in model.parameters():
    param.requires_grad = False

fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, NUM_OBJECTS + NUM_COLORS)
)
model = model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training
for epoch in range(1000):
    model.train()
    for i, (images, (obj_labels, color_labels)) in enumerate(train_loader):
        images = images.to(device)
        obj_labels = obj_labels.to(device)
        color_labels = color_labels.to(device)

        # Forward pass
        outputs = model(images)
        obj_outputs, color_outputs = outputs.split([NUM_OBJECTS, NUM_COLORS], dim=1)

        obj_loss = loss_func(obj_outputs, obj_labels)
        color_loss = loss_func(color_outputs, color_labels)

        # Total loss
        loss = obj_loss + color_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i, (images, (obj_labels, color_labels)) in enumerate(val_loader):
            images = images.to(device)
            obj_labels = obj_labels.to(device)
            color_labels = color_labels.to(device)

            outputs = model(images)
            obj_outputs, color_outputs = outputs.split([NUM_OBJECTS, NUM_COLORS], dim=1)

            obj_loss = loss_func(obj_outputs, obj_labels)
            color_loss = loss_func(color_outputs, color_labels)

            # Total loss
            loss = obj_loss + color_loss
            total_loss += loss.item()
        
        print('Validation loss:', total_loss / len(val_loader))

save_path = 'resnet18_multi_task_trained_classific.pth'
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
