#!/usr/bin/env python

"""Implementation of Person detector using MobileNet SSD on OpenCV DNN"""

# For Python2/3 compatibility
from __future__ import print_function
from __future__ import division

import sys
import os
import math
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
#from ros_pose.msg import BodyPose

import cv2
#print(cv2.__version__)

import numpy as np

__author__ = "Eric Dortmans"

class PersonDetection:
    """This class processes ROS Images using OpenCV DNN

    """

    def __init__(self):
        self.process_image_setup()
        self.bridge = CvBridge()
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
        self.network = rospy.get_param('~network', rospack.get_path('ros_pose')+'/config/'+'MobileNetSSD_deploy.prototxt')
        self.weights = rospy.get_param('~weights', rospack.get_path('ros_pose')+'/config/'+'MobileNetSSD_deploy.caffemodel')
        self.threshold = rospy.get_param('~threshold', 0.5) # Confidence threshold
        self.inWidth = rospy.get_param('~width', 300) # Resize input to this width
        self.inHeight = rospy.get_param('~height', 300) # Resize input to this height
        
        self.classNames = { 0: 'background',
                1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                14: 'motorbike', 15: 'person', 16: 'pottedplant',
                17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
        
        self.net = cv2.dnn.readNetFromCaffe(self.network, self.weights)
            
    def process_image(self):
        """Process the image using OpenCV DNN

        This code is run for reach image
        """           
        
        # size of original image
        imageWidth = self.image.shape[1]
        imageHeight = self.image.shape[0]
        
        # resizing, mean subtraction, normalizing, and channel swapping of the image
        image_resized = cv2.resize(self.image, (self.inWidth, self.inHeight))
        self.net.setInput(cv2.dnn.blobFromImage(image_resized, 0.007843, (self.inWidth, self.inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))

        # prediction
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] #Confidence of prediction 
            if confidence > self.threshold: # Only predictions above threshold 
                class_id = int(detections[0, 0, i, 1]) # Class label
                if class_id == 15: # persons
                  # Object location 
                  xLeftBottom = int(detections[0, 0, i, 3] * self.inWidth) 
                  yLeftBottom = int(detections[0, 0, i, 4] * self.inHeight)
                  xRightTop   = int(detections[0, 0, i, 5] * self.inWidth)
                  yRightTop   = int(detections[0, 0, i, 6] * self.inHeight)
                  
                  # Factor for scale to original size of image
                  heightFactor = imageHeight/float(self.inHeight) 
                  widthFactor = imageWidth/float(self.inWidth)
                  # Scale object detection to image
                  xLeftBottom = int(widthFactor * xLeftBottom) 
                  yLeftBottom = int(heightFactor * yLeftBottom)
                  xRightTop   = int(widthFactor * xRightTop)
                  yRightTop   = int(heightFactor * yRightTop)
                  
                  if self.display:
                    # Draw location of object  
                    cv2.rectangle(self.image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))
                    # Draw label and confidence of prediction in image resized
                    if class_id in self.classNames:
                        label = self.classNames[class_id] + ": " + str(confidence)
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        yLeftBottom = max(yLeftBottom, labelSize[1])
                        cv2.rectangle(self.image, (xLeftBottom, yLeftBottom - labelSize[1]),
                                             (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                             (255, 255, 255), cv2.FILLED)
                        cv2.putText(self.image, label, (xLeftBottom, yLeftBottom),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                        #print(label) #print class and confidence

        cv2.imshow("image", self.image)
        cv2.waitKey(1)


def main(args):
    rospy.init_node('pose_estimator', anonymous=True)
    ip = PersonDetection()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
