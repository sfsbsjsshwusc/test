#!/usr/bin/env python3
# coding=utf-8

import rospy
import cv2
from servo_control import send_servo_command
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from my_servo_control.msg import ServoCommand
# 彩色图像回调函数
def Cam_RGB_Callback(msg):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("格式转换错误: %s", e)
        return

    # 弹出窗口显示图片
    cv2.imshow("RGB", cv_image)
    cv2.waitKey(1)

# 主函数
if __name__ == "__main__":
    rospy.init_node("demo_cv_image")
    # 订阅机器人视觉传感器Kinect2的图像话题
    rgb_sub = rospy.Subscriber("/usb_cam/image_raw",Image,Cam_RGB_Callback,queue_size=10)
    # servo = ServoCommand()
    # pub = rospy.Publisher('servo_command', ServoCommand, queue_size=10)
    # send_servo_command(0,90)
    rospy.spin()
