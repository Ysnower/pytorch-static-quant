from PIL import Image
import onnxruntime as ort
import numpy as np


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result)).tolist()


if __name__ == "__main__":
    onnx_model_path = "./your_out.onnx"
    ort_session = ort.InferenceSession(onnx_model_path)
    # 输入输出层名字(固定写法)
    onnx_input_name = ort_session.get_inputs()[0].name
    # 输出层名字,可能有多个
    onnx_outputs_names = ort_session.get_outputs()[0].name
    img = Image.open('test.jpg').convert("L")
    img = img.resize((128, 128), 0)
    img = np.asarray(img, np.float32)/255.0
    img = img[np.newaxis, np.newaxis, :, :]
    input_blob = np.array(img, dtype=np.float32)
    onnx_result = ort_session.run([onnx_outputs_names], input_feed={onnx_input_name: input_blob})
    res = postprocess(onnx_result)
    idx = np.argmax(res)
    print("识别结果", idx)