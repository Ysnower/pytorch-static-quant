import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision
from torchvision import transforms
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import get_default_qconfig
from torch import optim
import os
import time
from utils import load_data
from mobilenetv3copy import MobileNetV3_Large


def evaluate_model(model, test_loader, device, criterion=None):
    model.eval()
    model.to(device)
    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0
        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)
    return eval_loss, eval_accuracy


def quant_fx(model, data_loader):
    # model.eval()
    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.eval()
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}
    prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
    print("开始校准")
    calibrate(prepared_model, data_loader)
    print("校准完毕")
    quantized_model = convert_fx(prepared_model)
    return quantized_model


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image.to('cpu'))


if __name__ == "__main__":
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
    model = MobileNetV3_Large(2)
    train_loader, test_loader = load_data(64, 8)
    # quantization
    state_dict = torch.load('./1.pth')  # 输入需要量化的模型
    model.load_state_dict(state_dict)
    model.to('cpu')
    model.eval()
    quant_model = quant_fx(model, train_loader)
    quant_model.eval()

    # eval
    print("开始验证")
    eval_loss, eval_accuracy = evaluate_model(model=quant_model,
                                              test_loader=test_loader,
                                              device=cpu_device,
                                              criterion=nn.CrossEntropyLoss())
    print("Epoch: {:02d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
        -1, eval_loss, eval_accuracy))

    torch.jit.save(torch.jit.script(quant_model), 'outQuant.pth')  # 保存量化模型outQuant.pth

    # 推理
    loaded_quantized_model = torch.jit.load('outQuant.pth')  # 加载量化模型后再测试一下准确性
    eval_loss, eval_accuracy = evaluate_model(model=loaded_quantized_model,
                                              test_loader=test_loader,
                                              device=cpu_device,
                                              criterion=nn.CrossEntropyLoss())
    print("Epoch: {:02d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
        -1, eval_loss, eval_accuracy))

