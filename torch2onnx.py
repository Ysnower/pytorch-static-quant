#  torch2onnx.py
import torch
import torchvision
from mobilenetv3copy import MobileNetV3_Large   # 引入模型

torch.set_grad_enabled(False)
torch_model = MobileNetV3_Large(2)  # 初始化网络
torch_model.load_state_dict(torch.load('./your_input.pth'), False)  # 加载训练好的pth模型
batch_size = 1  # 批处理大小
input_shape = (1, 128, 128)  # 输入数据,我这里是灰度训练所以1代表是单通道，RGB训练是3，128是图像输入网络的尺寸

# set the model to inference mode
torch_model.eval().cpu()  # cpu推理

x = torch.randn(batch_size, *input_shape).cpu()  # 生成张量
export_onnx_file = "./your_out.onnx"  # 要生成的ONNX文件名
torch.onnx.export(torch_model,
                  x,
                  export_onnx_file,
                  opset_version=10,
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=["input"],  # 输入名
                  output_names=["output"],  # 输出名
                  dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                                "output": {0: "batch_size"}})
