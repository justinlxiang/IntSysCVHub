import torch
from torchvision import models, transforms
from PIL import Image
import os

# Define paths
detected_images_path = '../../detectors/rcnn/Detected_Images'
color_weights_path = '../model_weights/real_color_weights.pt'
shape_weights_path = '../model_weights/real_shape_weights.pt'

# Load models
color_model = models.resnet18()
color_model.fc = torch.nn.Linear(color_model.fc.in_features, 7)  # 7 color classes
color_model.load_state_dict(torch.load(color_weights_path))
color_model.eval()

shape_model = models.resnet18()
shape_model.fc = torch.nn.Linear(shape_model.fc.in_features, 3)  # Assuming 4 shape classes
shape_model.load_state_dict(torch.load(shape_weights_path))
shape_model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define classes
color_classes = ['Black', 'Blue', 'Green', 'Orange', 'Purple', 'Red', 'White']
shape_classes = ['Circle', 'Quarter Circle', 'Rectangle', 'Semi Circle']

# Function to classify images
def classify_images(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict color
    color_outputs = color_model(image)
    _, color_pred = torch.max(color_outputs, 1)
    color = color_classes[color_pred.item()]

    # Predict shape
    shape_outputs = shape_model(image)
    _, shape_pred = torch.max(shape_outputs, 1)
    shape = shape_classes[shape_pred.item()]

    return color, shape

# Process images
for image_name in os.listdir(detected_images_path):
    image_path = os.path.join(detected_images_path, image_name)
    color, shape = classify_images(image_path)
    print(f'Image: {image_name}, Color: {color}, Shape: {shape}')
