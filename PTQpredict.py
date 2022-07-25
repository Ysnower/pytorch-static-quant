import time
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def predict():
    device = torch.device('cpu')
    model = torch.jit.load('outQuant.pth')
    model.eval()
    model.to(device)

    for i in range(1000):
        img = Image.open('1.jpg').convert("L")  # 输入图像推理，可以读取test.txt来验证准确性
        img = img.resize((128, 128), 0)
        img = transform_data(img).unsqueeze(0)
        img = img.to(device)
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        result = predicted[0].item()
        print(result)

if __name__ == "__main__":
    transform_data = transforms.Compose([transforms.Resize([128,128]),
                                        transforms.ToTensor()])
    predict()