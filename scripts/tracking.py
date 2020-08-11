#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
#import cv2 as cv
import rospy
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes,BoundingBox
import cv2
import glob
from opencv_apps.msg import LineArrayStamped
from std_msgs.msg import String
#from opencv_apps.msg import LineArrayStamped

from opencv_apps.msg import Line



class subscribers():

    def __init__(self):
        self.pub_lines = rospy.Publisher('/boundary_lines', LineArrayStamped, queue_size=1)

        self.pub = rospy.Publisher('/tracking', Image, queue_size=1)
        self.first_frame = True
        rospy.Subscriber("cropped_x_y", String, self.cropped_x_y, queue_size = 10)

        rospy.Subscriber("/four_hough_corners", String, self.four_corners_message, queue_size = 1)
#        rospy.Subscriber("cropped_x_y", String, self.cropped_x_y, queue_size = 1)


    def cropped_x_y(self,data):
        #print("CAME IN HERE")
        cropped_x_y = data.data
        self.cropped_x = cropped_x_y.split(',')[0]
        self.cropped_y = cropped_x_y.split(',')[1]
        #print(cropped_x,cropped_y)

    def four_corners_message(self,points):
        self.first_frame = True
        self.corners = points.data

        self.point1 = self.corners.split('/')[0]
        #print(self.point1)
        self.point2 = self.corners.split('/')[1]
        self.point3 = self.corners.split('/')[2]
        self.point4 = self.corners.split('/')[3]
        self.tracknow = True
        self.point1_x = float(self.point1.split(',')[0]) + float(self.cropped_x)
        self.point1_y = float(self.point1.split(',')[1]) + float(self.cropped_y)
        self.point2_x = float(self.point2.split(',')[0]) + float(self.cropped_x)
        self.point2_y = float(self.point2.split(',')[1]) + float(self.cropped_y)
        self.point3_x = float(self.point3.split(',')[0]) + float(self.cropped_x)
        self.point3_y = float(self.point3.split(',')[1]) + float(self.cropped_y)
        self.point4_x = float(self.point4.split(',')[0]) + float(self.cropped_x)
        self.point4_y = float(self.point4.split(',')[1]) + float(self.cropped_y)
        print(self.point1_x,self.point1_y)
        print(self.point2_x,self.point2_y)
        print(self.point3_x,self.point3_y)
        print(self.point4_x,self.point4_y)
        rospy.Subscriber("/rec_image", Image, self.callback_image, queue_size = 1)





    def callback_image(self,image):
        self.header = image.header
        
        #final_list = []

        bridge = CvBridge()
        self.tracking_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        if self.first_frame == True:
            '''
            self.feature_params = dict( maxCorners = 100,
                                   qualityLevel = 0.3,
                                   minDistance = 7,
                                   blockSize = 7 )
            '''
            # Parameters for lucas kanade optical flow
            
            self.lk_params = dict( winSize  = (15,15),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            self.color = np.random.randint(0,255,(100,3))

            # Take first frame and find corners in it
            self.old_frame = self.tracking_image
            self.old_gray = cv2.cvtColor(self.tracking_image, cv2.COLOR_BGR2GRAY)

            '''TEST FOR SHITOMASHI'''
            #self.arr = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)


            self.arr = np.array([[[self.point1_x, self.point1_y]],
                                [[self.point2_x, self.point2_y]],
                                [[self.point3_x, self.point3_y]],
                                [[self.point4_x, self.point4_y]]])
            '''
            self.arr = np.array([[[108., 69.]],
            [[261., 100.]],
            [[276., 177.]],
            [[59., 135.]]])
            '''
            #'''
            self.arr = np.float32(np.asarray(self.arr))
            print(self.arr)
            # Create a mask image for drawing purposes
            self.mask = np.zeros_like(self.old_frame)

            self.first_frame = False
        else:
            #ret,frame = cap.read()
            self.frame = self.tracking_image
            self.frame_gray = cv2.cvtColor(self.tracking_image, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            self.p1, self.st, self.err = cv2.calcOpticalFlowPyrLK(self.old_gray, self.frame_gray, self.arr, None, **self.lk_params)

            # Select good points
            self.good_new = self.p1[self.st==1]
            self.good_old = self.arr[self.st==1]

            # draw the tracks
            final_list = []
            for i,(new,old) in enumerate(zip(self.good_new, self.good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                self.mask = cv2.line(self.mask, (a,b),(c,d), self.color[i].tolist(), 2)
                self.frame = cv2.circle(self.frame,(a,b),5,self.color[i].tolist(),-1)
                final_list.append([int(a),int(b)])
            self.img = cv2.add(self.frame,self.mask)

            #cv2.imshow('frame',self.img)
            #k = cv2.waitKey(30) & 0xff
            #if k == 27:
            #    "PRESSED K"

            # Now update the previous frame and previous points
            self.old_gray = self.frame_gray.copy()
            self.arr = self.good_new.reshape(-1,1,2)
        

        print('point positions',final_list)

        message = LineArrayStamped()
        #for i in range(4):
        message.lines.append([])
        message.lines[0] = Line()
        message.lines[0].pt1.x = final_list[0][0]
        message.lines[0].pt1.y = final_list[0][1]
        message.lines[0].pt2.x = final_list[1][0]
        message.lines[0].pt2.y = final_list[1][1]

        message.lines.append([])
        message.lines[1] = Line()
        message.lines[1].pt1.x = final_list[1][0]
        message.lines[1].pt1.y = final_list[1][1]
        message.lines[1].pt2.x = final_list[2][0]
        message.lines[1].pt2.y = final_list[2][1]

        message.lines.append([])
        message.lines[2] = Line()
        message.lines[2].pt1.x = final_list[2][0]
        message.lines[2].pt1.y = final_list[2][1]
        message.lines[2].pt2.x = final_list[3][0]
        message.lines[2].pt2.y = final_list[3][1]
        
        message.lines.append([])
        message.lines[3] = Line()
        message.lines[3].pt1.x = final_list[3][0]
        message.lines[3].pt1.y = final_list[3][1]
        message.lines[3].pt2.x = final_list[0][0]
        message.lines[3].pt2.y = final_list[0][1]



        message.header = self.header
        self.pub_lines.publish(message)

        self.tracking_image = bridge.cv2_to_imgmsg(self.img, encoding='bgr8')
        self.tracking_image.header = self.header
        self.pub.publish(self.tracking_image)



def main():
    rospy.init_node('optical_flow')

    #pub = Publishers()
    sub = subscribers()

    rospy.spin()

if __name__ == '__main__':
    main()