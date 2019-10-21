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
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import Float32

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
        self.image_publisher = rospy.Publisher("image_roi", Image, queue_size=1)
        self.roi_publisher = rospy.Publisher("roi", RegionOfInterest, queue_size=1)
        self.velocity_publisher = rospy.Publisher("velocity", Float32, queue_size=1)
               
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
        self.timestamp = image_msg.header.stamp
        self.process_image()

    def process_image_setup(self):
        """Setup for image processing. 

        This code will run only once to setup image processing.
        """
        import rospkg
        rospack = rospkg.RosPack()
        
        # Setup parameters
        self.display = rospy.get_param('~display', True)
        self.record = rospy.get_param('~record', False)
        self.network = rospy.get_param('~network', rospack.get_path('ros_pose')+'/config/'+'MobileNetSSD_deploy.prototxt')
        self.weights = rospy.get_param('~weights', rospack.get_path('ros_pose')+'/config/'+'MobileNetSSD_deploy.caffemodel')
        self.threshold = rospy.get_param('~threshold', 0.8) # Confidence threshold
        self.inWidth = rospy.get_param('~width', 300) # Resize input to this width
        self.inHeight = rospy.get_param('~height', 300) # Resize input to this height
 
        # Setup Deep Neural Network for person detection
        """       
        self.classNames = { 0: 'background',
                1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                14: 'motorbike', 15: 'person', 16: 'pottedplant',
                17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
        """
        self.classNames = { 15: 'person' }
                
        self.net = cv2.dnn.readNetFromCaffe(self.network, self.weights)
        #self.net = cv2.dnn.readNetFromTensorflow(self.weights, self.network)
     
        # Setup Regin-Of-Interest estimation
        self.alfa = 0.9 # smoothing factor
        self.roi_last = None # previous roi

        # Setup velocity estimation
        self.previous_timestamp = None
        self.previous_height = None
        self.velocity = 0

        # Setup video writer 
        if self.record:
          # Define the codec and create VideoWriter object
          fourcc = cv2.VideoWriter_fourcc(*'XVID')
          self.video = cv2.VideoWriter('person.avi', fourcc, 15, (640, 480), True)


    def process_image(self):
        """Process the image using OpenCV DNN

        This code is run for reach image
        """
        
        # Copy of image for display purposes
        display_img = self.image.copy()       
                
        # Size of original image
        imageWidth = self.image.shape[1]
        imageHeight = self.image.shape[0]
        
        # Resizing, mean subtraction, normalizing, and channel swapping of the image
        self.net.setInput(cv2.dnn.blobFromImage(self.image, 0.007843, 
          (self.inWidth, self.inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))

        # Detect various object classes
        detections = self.net.forward()
        
        # Estimate ROI of person we are focussing on
        biggest_size = 0
        roi = [0, 0, imageHeight, imageWidth]
        for detection in detections[0, 0, :, :]:
            class_id = detection[1]
            confidence = detection[2]
            if confidence > self.threshold: # Only predictions above threshold
                if class_id in self.classNames:
                  # Object location 
                  xLeftTop = int(detection[3] * self.inWidth) 
                  yLeftTop = int(detection[4] * self.inHeight)
                  xRightBottom   = int(detection[5] * self.inWidth)
                  yRightBottom   = int(detection[6] * self.inHeight)          
                  # Factor for scale to original size of image
                  heightFactor = imageHeight/float(self.inHeight) 
                  widthFactor = imageWidth/float(self.inWidth)
                  # Scale object detection to image
                  xLeftTop = int(widthFactor * xLeftTop) 
                  yLeftTop = int(heightFactor * yLeftTop)
                  xRightBottom = int(widthFactor * xRightBottom)
                  yRightBottom = int(heightFactor * yRightBottom)
                                   
                  # Remember ROI of biggest one
                  size = (xRightBottom - xLeftTop) * (yRightBottom - yLeftTop)
                  margin = 10
                  if size > biggest_size:
                    biggest_size = size
                    roi_x = max(0, xLeftTop - margin)
                    roi_y = max(0, yLeftTop - margin)
                    roi_height = min(imageHeight, yRightBottom - yLeftTop + 2*margin)
                    roi_width = min(imageWidth, xRightBottom - xLeftTop + 2*margin)
                    roi = [roi_x, roi_y, roi_height, roi_width]
                                           
                  if self.display:
                    # Draw location of object  
                    cv2.rectangle(display_img, (xLeftTop, yLeftTop), (xRightBottom, yRightBottom), (0, 255, 0))                   
                    # Draw label and confidence of prediction
                    #label = self.classNames[class_id] + ": " + str(confidence)
                    label = str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    yLeftTop = max(yLeftTop, labelSize[1])
                    #cv2.rectangle(display_img, (xLeftTop, yLeftTop - labelSize[1]),
                    #              (xLeftTop + labelSize[0], yLeftTop + baseLine),
                    #              (255, 255, 255), cv2.FILLED)
                    cv2.putText(display_img, label, (xLeftTop, yLeftTop),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))    

        # Filter ROI using exponential moving average
        if self.roi_last is not None:
          roi = [int(self.alfa * i + (1-self.alfa) * j) for i,j in zip(self.roi_last,roi)]
        self.roi_last = roi

        # Publish ROI
        roi_msg = RegionOfInterest()
        roi_msg.x_offset = roi[0] # Leftmost pixel of the ROI
        roi_msg.y_offset = roi[1] # Topmost pixel of the ROI
        roi_msg.height = roi[2]   # Height of ROI
        roi_msg.width = roi[3]    # Width of ROI
        self.roi_publisher.publish(roi_msg)   
        roi_image = self.image[roi[1]:roi[1]+roi[2], roi[0]:roi[0]+roi[3]].copy()    
        self.image_publisher.publish(self.to_imgmsg(roi_image))     

        # Estimate velocity of person
        if self.previous_timestamp == None: # first image
          restart = True
        else:
          dt = (self.timestamp - self.previous_timestamp).to_sec()
          if dt < 0: # loop restart
            print("== loop restart ==")
            restart = True
          elif dt > 0.3: # time to calculate
            #print("dt: ",dt)
            dx = roi[2] - self.previous_height
            self.velocity = dx / dt
            restart = True
          else: # just wait for time to pass
            restart = False       
        if restart:
          self.previous_timestamp = self.timestamp
          self.previous_height = roi[2]

        # Publish velocity
        velocity_msg = Float32()
        velocity_msg.data = self.velocity
        self.velocity_publisher.publish(velocity_msg)

        if self.display:
          # Show Region Of Interest
          cv2.rectangle(display_img, (roi[0], roi[1]), (roi[0]+roi[3], roi[1]+roi[2]), (0, 0, 255))
          #cv2.rectangle(display_img, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 0, 255))
          
          # Show velocity
          #label = str(self.velocity)
          label = "{0:.2f}".format(self.velocity)
          labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
          cv2.putText(display_img, label, (roi[0], roi[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
          
          # Show images
          cv2.imshow("image", display_img)
          cv2.imshow("roi", roi_image)       
          cv2.waitKey(1)

        if self.record:
          # Write video frame
          self.video.write(display_img)


def main(args):
    rospy.init_node('person_detector', anonymous=True)
    ip = PersonDetection()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
