# -*- coding: utf-8 -*-
import cv2
import sys
from PIL import Image
import numpy as np

import extremenetClientModule

cameraID = 59

timeStamps = 121
img = cv2.imread("664.jpg")
# 测试AlgorithmGetProperty函数
isTrue, GPUUseup, alg_name = extremenetClientModule.AlgorithmGetProperty()
print("isTrue:",isTrue)
print("GPUUseup:",GPUUseup)
print("alg_name:",alg_name)

# 测试AlgorithmInitModle函数
model_args = '{"test_k":"test_value"}'
address = 'localhost'
port = '8001'
roi = '[]'
isTrue, inference = extremenetClientModule.AlgorithmInitModle(0, './extremenetClientModuleLogs', model_args=model_args, address=address, port=port, roi=roi)
print("isTrue:",isTrue)

# 测试AlgorithmRun函数
ret, result, detection_result_json, save_input, save_output =extremenetClientModule.AlgorithmRun(cameraID, timeStamps, img,inference)
print("ret:",ret)
print("result:",result)
print("detection_result_json:",detection_result_json)