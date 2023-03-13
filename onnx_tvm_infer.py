import time
import numpy as np
import tvm
import tvm.relay as relay
from tvm.contrib import graph_runtime
import cv2


def preprocess(img):
    h, w = img.shape[:2]
    if h < w:
        distance = int((w - h) / 2)
        img = cv2.copyMakeBorder(img, distance, distance, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        distance = int((h - w) / 2)
        img = cv2.copyMakeBorder(img, 0, 0, distance, distance, cv2.BORDER_CONSTANT, value=0)
    img = cv2.resize(img, (128, 128), cv2.LINE_AA)
    img = img.astype(np.float32) / 255.0
    img = img[np.newaxis, np.newaxis, :, :]
    img = np.array(img, dtype=np.float32)
    return img


if __name__ == "__main__":
    test_json = "test.json"
    test_lib = "test.so"
    test_params = "test.params"
    loaded_json = open(test_json).read()
    loaded_lib = tvm.runtime.load_module(test_lib)
    loaded_params = bytearray(open(test_params, 'rb').read())

    img = cv2.imread("test.jpg", 0)
    img_input = preprocess(img)
    ctx = tvm.cpu(0)
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    start = time.time()
    for i in range(1000):
        module.set_input("input", img_input)
        module.run()
        out_deploy = module.get_output(0).numpy()
        out = np.argmax(out_deploy)
        # print(out_deploy, out)
    print(time.time()-start)

