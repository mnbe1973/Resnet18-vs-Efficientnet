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
def predict_object_and_color(image_path, model_path='resnet18_multi_task_trained_classific.pth'):
    # Define the transform for the input image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image and apply transforms
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    
    # Load the trained model
    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, NUM_OBJECTS + NUM_COLORS),
        nn.ReLU(),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Predict
    with torch.no_grad():
        output = model(img)
        probs = torch.exp(output)
        obj_probs, color_probs = probs.split([NUM_OBJECTS, NUM_COLORS], dim=1)

        predicted_obj = torch.argmax(obj_probs).item()
        predicted_color = torch.argmax(color_probs).item()

    return predicted_obj, predicted_color

# Example usage:
img_path = 'ob4_blue.png'
predicted_obj, predicted_color = predict_object_and_color(img_path)
print(f"Predicted Object: {predicted_obj}")
print(f"Predicted Color: {predicted_color}")
