# -*- coding: utf-8 -*-
import numpy as np

# ######
# obj_detection
# ######
# 种类名文本路径
labels_url = '/home/jetson/tensorrtx-yolov5-v6.0/yolov5/class/labels_znjy.txt'
# trt plugins路径
plugin_url = "/home/jetson/tensorrtx-yolov5-v6.0/yolov5/build/libmyplugins.so"
# trt engine路径
engine_url = "/home/jetson/tensorrtx-yolov5-v6.0/yolov5/build/znjy_v3.0.engine" #ruler:改成自己配置的了

# yolo 置信度阈值
CONF_THRESH = 0.70
# yolo 交并比阈值
IOU_THRESHOLD = 0.5
            
# #####################################################################
