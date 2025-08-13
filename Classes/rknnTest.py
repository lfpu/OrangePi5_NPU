import numpy as np
import cv2
from rknn.api import RKNN
import time

modelPath = 'model.onnx'  # 假设模型路径为'rknn_model.rknn'

rknn = RKNN()
rknn.config (mean_values=[[127.5, 127.5, 127.5]], std_values=[[128, 128, 128]])
rknn.build(do_quantization=False)
# 构建RKNN模型
# 加载RKNN模型
ret = rknn.load_onnx(modelPath)
if ret != 0:
    print('Load RKNN model failed!')
    exit(ret)
# 配置输入输出
ret =rknn.init_runtime()
if ret != 0:
    print('Init runtime failed!')
    exit(ret)
# 读取图片
image_path = 'test.jpg'  # 假设测试图片路径为'test_image.jpg'
image = cv2.imread(image_path)
if image is None:
    print('Failed to read image!')
    exit(1)
# 预处理图片
image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
image = cv2.resize(image, (640, 640))  # 假设模型
# 输入尺寸为640x640
outputs=rknn.inference(inputs=[image])
if outputs is None:
    print('Inference failed!')
    exit(1)
print(outputs.shape)  # 输出结果的形状