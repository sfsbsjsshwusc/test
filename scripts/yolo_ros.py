#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import ctypes
import sys
import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import rospy
import cv2
from sensor_msgs.msg import Image
from user_msgs.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge, CvBridgeError

try:
    from params import labels_url, plugin_url, engine_url, CONF_THRESH, IOU_THRESHOLD
except ModuleNotFoundError:
    sys.path.append('/home/jetson/catkin_ws/algorithm/detection/scripts/')
    from params import labels_url, plugin_url, engine_url, CONF_THRESH, IOU_THRESHOLD
    sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

INPUT_W = 640
INPUT_H = 640

# load labels
categories = []
with open(labels_url, encoding='utf-8') as file:
    class_num = int(file.readline())
    for line in file:
        label = line.strip()
        if label == '':
            continue
        categories.append(label)

if not categories:
    print('\nFail to load labels, please check again!\n')
    exit()
elif len(categories) != class_num:
    print('\nclass number conflit, please check again!\n')
    exit()


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('000000', '0096FF', '0000FF', '89CFF0', '454B1B', '36454F', 'FF0000', 'E35335', 'A52A2A', 'FFEA00',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

# draw boxes
def plot_boxes(src, result_boxes, result_scores, result_classid):
    colors = Colors()

    for i in range(len(result_boxes)):
        c1 = (int(result_boxes[i][0]), int(result_boxes[i][1]))
        c2 = (int(result_boxes[i][2]), int(result_boxes[i][3]))  # conor point
        color = colors(result_classid[i], True)
        cv2.rectangle(src, c1, c2, color, thickness=3, lineType=cv2.LINE_AA)
        if result_classid is not None:
            txt = "{}:{:.2f}".format(categories[int(result_classid[i])], result_scores[i])
            t_size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.5, thickness=2)
            c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(src, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(src,
                        text=txt,
                        org=(c1[0], c1[1] - 2),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=[225, 255, 255],
                        thickness=2)

    return src


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def __del__(self):
        print("delete object to release memory")

    def infer(self, input_image_path):
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(input_image_path)
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        result_boxes, result_scores, result_classid = self.post_process(output, origin_h, origin_w)

        return image_raw, result_boxes, result_scores, result_classid

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def preprocess_image(self, image_raw):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)

        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        # y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def nms(self, boxes, scores, iou_threshold=IOU_THRESHOLD):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = scores
        keep = []
        index = scores.argsort()[::-1]
        while index.size > 0:
            i = index[0]  # every time the first is the biggst, and add it directly
            keep.append(i)

            x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
            h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

            idx = np.where(ious <= iou_threshold)[0]
            index = index[idx + 1]  # because index start from 1

        return keep

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        boxes = pred[:, :4]
        scores = pred[:, 4]
        classid = pred[:, 5]
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        indices = self.nms(boxes, scores, IOU_THRESHOLD)
        result_boxes = boxes[indices, :]
        result_scores = scores[indices]
        result_classid = classid[indices]

        return result_boxes, result_scores, result_classid


def load_model():
    # load custom plugin and engine
    try:
        ctypes.CDLL(plugin_url)
        wrapper = YoLov5TRT(engine_url)
    except OSError:
        print("请检查engine文件")
        exit()

    return wrapper

# 彩色图像回调函数
def process_frame(frame):
    boundingBoxes = BoundingBoxes()
    boundingBoxes.header.stamp = rospy.Time.now()
    getImageStatus = True

    t0 = time.time()
    # 推理
    img, result_boxes, result_scores, result_classid = yolov5_wrapper.infer(frame)  # 推理返回结果
    fps = 1 / (time.time() - t0)
    num = len(result_classid)

    for i in range(num):
        boundingBox = BoundingBox()
        boundingBox.probability = float(result_scores[i])
        boundingBox.xmin = int(result_boxes[i][0])
        boundingBox.ymin = int(result_boxes[i][1])
        boundingBox.xmax = int(result_boxes[i][2])
        boundingBox.ymax = int(result_boxes[i][3])
        boundingBox.Class = str(result_classid[i])
        boundingBoxes.bounding_boxes.append(boundingBox)

    # 发布检测框信息
    det_pub.publish(boundingBoxes)
    
    # 画出检测框并显示FPS
    # img = plot_boxes(img, result_boxes, result_scores, result_classid)
    # cv2.putText(img, f'FPS:{int(fps)}', (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (125, 0, 255), 2)
    # cv2.imshow("result", img)
    # cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node("yolo_ros")
    
    yolov5_wrapper = load_model()  # 加载模型
    print("加载完成")

    # 发布yolo_detection msg
    det_pub = rospy.Publisher('/yolo/detection', BoundingBoxes, queue_size=6)  # 改频率

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("摄像头打开失败")
        exit()

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break
        process_frame(frame)
        
        # if cv2.waitKey(1) & 0xFF == 27:
        #     print("关闭小车眼睛")
        #     break

    cap.release()
    # cv2.destroyAllWindows()
    
    yolov5_wrapper.destroy()