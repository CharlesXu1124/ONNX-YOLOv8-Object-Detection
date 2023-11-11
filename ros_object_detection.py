#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import openai
import numpy as np
import os
import cv2
from yolov8 import YOLOv8
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from std_msgs.msg import Header
from cv_bridge import CvBridge

import requests
from PIL import Image as PIL_Image
from numpy import asarray
import torch
import array
import math

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
        self.yolo_publisher = self.create_publisher(Float32MultiArray, 'yolo_result', 10)
        self.img_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.img_callback,
            10)

        self.depth_subscription = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            10)

        # Initialize YOLOv7 object detector
        self.model_path = "models/yolov8s.onnx"
        self.yolov8_detector = YOLOv8(self.model_path, conf_thres=0.5, iou_thres=0.5)
        self.get_logger().info("============Yolo Model Ready===========")
        self.bridge = CvBridge()
        self.depth_map = np.zeros((640, 480, 1), np.float16)

        # camera parameters
        self.cx = 319.5
        self.cy = 239.5
        self.fx = 570.3422047415297
        self.fy = 570.3422047415297

        # prevent variable not used warning
        self.img_subscription

    def add_point_to_pointcloud(self, point_cloud, x, y, z):
        point = [x, y, z]
        point_cloud.append(point)
        return point_cloud

    def img_callback(self, Image):
        self.get_logger().info("image received")
        cv_image = self.bridge.imgmsg_to_cv2(Image, desired_encoding='passthrough')
        boxes, scores, labels = self.yolov8_detector(cv_image)
        results = ""

        yolo_msg = Float32MultiArray()
        
        # Create a PointCloud2 message
        header = Header()
        header.frame_id = 'base_link'  # Change this frame_id as needed
        points = []

        
        for box, score, label in zip(boxes, scores, labels):
            center_x = math.floor((box[0] + box[2]) / 2.0)
            center_y = math.floor((box[1] + box[3]) / 2.0)
            depth_xy = self.depth_map[center_x][center_y]
            # https://stackoverflow.com/questions/31265245/extracting-3d-coordinates-given-2d-image-points-depth-map-and-camera-calibratio
            # compute x, y, z of the detected object
            x = (center_x - self.cx) * self.depth_map[center_x][center_y] / self.fx
            y = (center_y - self.cy) * self.depth_map[center_x][center_y] / self.fy
            z = self.depth_map[center_x][center_y]

            points.append(float(x))
            points.append(float(y))
            points.append(float(z))
            points.append(float(label))

        yolo_msg.data = points
        print(points)

        self.yolo_publisher.publish(yolo_msg)            

    def depth_callback(self, Image):
        self.depth_map = self.bridge.imgmsg_to_cv2(Image, desired_encoding='passthrough') / 1000.0


def main(args=None):
    
    rclpy.init(args=args)
    RosYoloNode = RosYolov8Node()
    rclpy.spin(RosYoloNode)
    RosYoloNode.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()