#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from yolov8 import YOLOv8
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
from cv_bridge import CvBridge

from numpy import asarray
import math
import struct

class RosYolov8Node(Node):
    def __init__(self):
        super().__init__("ros_memeye_node")
        self.yolo_publisher = self.create_publisher(Float32MultiArray, '/yolo_result', 10)
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

        self.point_cloud_subscription = self.create_subscription(
            PointCloud2,
            "/camera/depth/points",
            self.points_callback,
            10
        )

        # Initialize YOLOv7 object detector
        self.model_path = "models/yolov8s.onnx"
        self.yolov8_detector = YOLOv8(self.model_path, conf_thres=0.5, iou_thres=0.5)
        self.get_logger().info("============Yolo Model Ready===========")
        self.bridge = CvBridge()
        self.point_cloud = PointCloud2()
        self.depth_map = np.zeros((480, 640), np.float32)

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
        cv_image = self.bridge.imgmsg_to_cv2(Image, desired_encoding='passthrough')
        boxes, scores, labels = self.yolov8_detector(cv_image)

        yolo_msg = Float32MultiArray()

        # Create a PointCloud2 message
        header = Header()
        header.frame_id = 'base_link'  # Change this frame_id as needed
        points = []
        
        if len(self.point_cloud.data) == 0:
            return

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            center_x = int((x1 + x2) / 2.0)
            center_y = int((y1 + y2) / 2.0)
            offset = center_y * 640 + center_x

            x = struct.unpack("f", bytes([self.point_cloud.data[offset * 16],
                 self.point_cloud.data[offset * 16 + 1],
                 self.point_cloud.data[offset * 16 + 2],
                 self.point_cloud.data[offset * 16 + 3]
            ]))

            y = struct.unpack("f", bytes([self.point_cloud.data[offset * 16 + 4],
                 self.point_cloud.data[offset * 16 + 5],
                 self.point_cloud.data[offset * 16 + 6],
                 self.point_cloud.data[offset * 16 + 7]
            ]))

            z = struct.unpack("f", bytes([self.point_cloud.data[offset * 16 + 8],
                 self.point_cloud.data[offset * 16 + 9],
                 self.point_cloud.data[offset * 16 + 10],
                 self.point_cloud.data[offset * 16 + 11]
            ]))

            if math.isnan(x[0]) or math.isnan(y[0]) or math.isnan(z[0]):
                continue

            points.append(float(x[0]))
            points.append(float(y[0]))
            points.append(float(z[0]))
            points.append(float(label))

        yolo_msg.data = points
        self.yolo_publisher.publish(yolo_msg)

    def points_callback(self, points):
        self.point_cloud = points

def main(args=None):
    rclpy.init(args=args)
    RosYoloNode = RosYolov8Node()
    rclpy.spin(RosYoloNode)
    RosYoloNode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()