#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import openai
import numpy as np
import os
import cv2
from yolov8 import YOLOv8
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import requests
from PIL import Image as PIL_Image
from numpy import asarray
import torch

class RosYolov8Node(Node):
    def __init__(self):
        super().__init__("ros_memeye_node")
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.yolo_publisher = self.create_publisher(String, 'yolo_result', 10)
        self.img_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.img_callback,
            10)

        # Initialize YOLOv7 object detector
        self.model_path = "models/yolov8s.onnx"
        self.yolov8_detector = YOLOv8(self.model_path, conf_thres=0.5, iou_thres=0.5)
        self.get_logger().info("============Yolo Model Ready===========")
        self.bridge = CvBridge()

        # prevent variable not used warning
        self.img_subscription


    def img_callback(self, Image):
        self.get_logger().info("image received")
        cv_image = self.bridge.imgmsg_to_cv2(Image, desired_encoding='passthrough')
        boxes, scores, labels = self.yolov8_detector(cv_image)
        results = ""

        for label in labels:
            results += " "
            results += self.class_names[label]
        print(results)



def main(args=None):
    rclpy.init(args=args)
    RosYoloNode = RosYolov8Node()
    rclpy.spin(RosYoloNode)
    RosYoloNode.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()