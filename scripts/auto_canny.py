#!/usr/bin/env python
# -*- coding: utf-8 -*-


import rospy
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from darknet_ros_msgs.msg import BoundingBoxes,BoundingBox
import cv2
import glob
from opencv_apps.msg import LineArrayStamped
from std_msgs.msg import String



class subscribers():

    def __init__(self):
        self.pub_edges = rospy.Publisher('/edges', Image, queue_size=1)
        self.pub_image = rospy.Publisher('/cropped_image', Image, queue_size=1)
        self.cropped_x_y = rospy.Publisher("/cropped_x_y", String, queue_size = 1)
        rospy.Subscriber("/rec_image", Image, self.callback_image, queue_size = 1,buff_size=2**24)
        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bounding_boxes_callback, queue_size = 1,buff_size=2**24)




    def bounding_boxes_callback(self, bbox):
        self.bbox_xmin = bbox.bounding_boxes[0].xmin
        self.bbox_ymin = bbox.bounding_boxes[0].ymin
        self.bbox_xmax = bbox.bounding_boxes[0].xmax
        self.bbox_ymax = bbox.bounding_boxes[0].ymax
        self.autocanny(self.image)

    #AUTOCANNY BASED ON ADRIAN ROSEBROCK POST (MEDIAN)
    def auto_canny(self, image, sigma=0.33):
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    #OTSU CANNY (OTSU BINARIZATION)
    def otsu_canny(self, image, lowrate=0.5):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Otsu's thresholding
        self.ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
        edged = cv2.Canny(image, threshold1=(self.ret * lowrate), threshold2=self.ret)

        # return the edged image
        return edged



    def callback_image(self,detection_image):

        self.detected_image = Image()
        self.edged_image = Image()

        bridge = CvBridge()
        self.image = bridge.imgmsg_to_cv2(detection_image, desired_encoding='bgr8')


    def autocanny(self,image):
        bridge = CvBridge()
        #crop original image to only have the bounding box as image, also add 10-20% more to make sure that all the object is inside, in case of a tight detection
        self.bigger_bbox = 0.1 #added percentage to increase size of bounding box and ensure object is inside
        height, width, _ = self.image.shape
        
        crop_ymin = int(self.bbox_ymin - (self.bigger_bbox * (self.bbox_ymax - self.bbox_ymin)) / 2)
        crop_ymax = int(self.bbox_ymax + (self.bigger_bbox * (self.bbox_ymax - self.bbox_ymin)) / 2)
        crop_xmin = int(self.bbox_xmin - (self.bigger_bbox * (self.bbox_xmax - self.bbox_xmin)) / 2)
        crop_xmax = int(self.bbox_xmax + (self.bigger_bbox * (self.bbox_xmax - self.bbox_xmin)) / 2)


        if crop_ymin < 0:
            crop_ymin = 0
        if crop_ymax > height:
            crop_ymax = height
        if crop_xmin < 0:
            crop_xmin = 0 
        if crop_xmax > width:
            crop_xmax = width



        self.crop_img = self.image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        self.gray = cv2.cvtColor(self.crop_img, cv2.COLOR_BGR2GRAY)
        self.auto = self.otsu_canny(self.gray)

        self.pub_edges.publish(bridge.cv2_to_imgmsg(self.auto, "mono8"))
        self.pub_image.publish(bridge.cv2_to_imgmsg(self.crop_img, "bgr8"))
        self.cropped_x_y.publish(str(crop_xmin) + ',' + str(crop_ymin))



def main():
    rospy.init_node('canny_auto')

    #pub = Publishers()
    sub = subscribers()

    rospy.spin()

if __name__ == '__main__':
    main()
