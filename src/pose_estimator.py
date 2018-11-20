#!/usr/bin/env python

"""Implementation of OpenPose using MobileNet on OpenCV DNN"""

# For Python2/3 compatibility
from __future__ import print_function
from __future__ import division

import sys
import os
import math
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ros_pose.msg import BodyPose

import cv2
#print(cv2.__version__)

import numpy as np

__author__ = "Eric Dortmans"

class PoseEstimation:
    """This class processes ROS Images using OpenCV DNN

    """

    def __init__(self):
        self.process_image_setup()
        self.bridge = CvBridge()
        self.pose_publisher = rospy.Publisher("body_pose", BodyPose, queue_size=1)
        self.image_subscriber = rospy.Subscriber("image_raw", Image, self.on_image_message,  
            queue_size=1, buff_size=2**24) # large buff_size for lower latency

    def to_cv2(self, image_msg):
        """Convert ROS image message to OpenCV image

        """
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except CvBridgeError as e:
            print(e)
        return image

    def to_imgmsg(self, image):
        """Convert OpenCV image to ROS image message

        """
        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        except CvBridgeError as e:
            print(e)
        return image_msg

    def on_image_message(self, image_msg):
        """Process received ROS image message
        """
        self.image = self.to_cv2(image_msg)
        self.process_image()

    def process_image_setup(self):
        """Setup for image processing. 

        This code will run only once to setup image processing.
        """
        import rospkg
        rospack = rospkg.RosPack()
        
        self.display = rospy.get_param('~display', True)
        self.network = rospy.get_param('~network', rospack.get_path('ros_pose')+'/config/'+'graph_opt.pb') # Trained Tensorflow network
        self.threshold = rospy.get_param('~threshold', 0.2) # Threshold value for pose parts heat map
        self.inWidth = rospy.get_param('~width', 368) # Resize input to this width
        self.inHeight = rospy.get_param('~height', 368) # Resize input to this height
        
        self.BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

        self.BODY_PARTS_REVERSED = {v:k for k,v in self.BODY_PARTS.items()}
        self.BODY_PART_NAMES = [v for k,v in self.BODY_PARTS_REVERSED.items()]
        #print(self.BODY_PART_NAMES)

        self.POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

        self.net = cv2.dnn.readNetFromTensorflow(self.network)
            
    def process_image(self):
        """Process the image using OpenCV DNN

        This code is run for reach image
        """           
        
        imageWidth = self.image.shape[1]
        imageHeight = self.image.shape[0]
        
        # resizing, mean subtraction, normalizing, and channel swapping of the image
        #image_resized = self.image
        image_resized = cv2.resize(self.image, (self.inWidth, self.inHeight))
        self.net.setInput(cv2.dnn.blobFromImage(image_resized, 1.0, (self.inWidth, self.inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        # prediction
        out = self.net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        pose_msg = BodyPose()
        pose_msg.header.stamp = rospy.Time.now()
        #pose_msg.part = self.BODY_PART_NAMES

        assert(len(self.BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(self.BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (imageWidth * point[0]) / out.shape[3]
            y = (imageHeight * point[1]) / out.shape[2]
            pose_msg.part.append(self.BODY_PART_NAMES[i])
            pose_msg.x.append(int(x))
            pose_msg.y.append(int(y))
            pose_msg.confidence.append(conf)
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > self.threshold else None)

        if self.display:
            for pair in self.POSE_PAIRS:
                partFrom = pair[0]
                partTo = pair[1]
                assert(partFrom in self.BODY_PARTS)
                assert(partTo in self.BODY_PARTS)

                idFrom = self.BODY_PARTS[partFrom]
                idTo = self.BODY_PARTS[partTo]
                if points[idFrom] and points[idTo]:
                    if self.display:
                        cv2.line(self.image, points[idFrom], points[idTo], (0, 255, 0), 3)
                        cv2.ellipse(self.image, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                        cv2.ellipse(self.image, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)         
            cv2.imshow("image", self.image)
            cv2.waitKey(1)

        self.pose_publisher.publish(pose_msg)

def main(args):
    rospy.init_node('pose_estimator', anonymous=True)
    ip = PoseEstimation()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
