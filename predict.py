import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNN

# Load model
model = CNN()
model.load_state_dict(torch.load("digit_cnn.pth"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

# Load image
img = Image.open("test.png")
img = transform(img)
img = img.unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(img)
    prediction = torch.argmax(output, dim=1)

print("Predicted digit:", prediction.item())