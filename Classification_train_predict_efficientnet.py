import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Example data
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

# Load EfficientNet-B0
model = models.efficientnet_b0(pretrained=True)

# Freeze pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier to output NUM_OBJECTS + NUM_COLORS
fc_inputs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(fc_inputs, NUM_OBJECTS + NUM_COLORS)
)
model = model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training
for epoch in range(100):
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
        
        print(f'Epoch [{epoch+1}/1000], Validation loss: {total_loss / len(val_loader):.4f}')

# Save the trained model
save_path = 'efficientnet_b0_multi_task_trained_classific.pth'
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
