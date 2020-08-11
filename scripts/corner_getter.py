#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import cv2
import numpy as np
import random 
import itertools
import time
from shapely.geometry import LineString
import rospy
from opencv_apps.msg import LineArrayStamped
from opencv_apps.msg import Line
from geometry_msgs.msg import Point
from numpy import ones,vstack
from std_msgs.msg import String
from sensor_msgs.msg import Image,CameraInfo
from operator import itemgetter
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial import distance as dist



class subscribe():

	def __init__(self):
		self.pub = rospy.Publisher('/connecting_lines', LineArrayStamped, queue_size=1)
		self.publisher = rospy.Publisher("/corner_points", Image, queue_size = 1)
		self.fourpoints = rospy.Publisher("/four_hough_corners", String, queue_size = 1)
		
		#rospy.Subscriber("/cropped_image", Image, self.callback_image, queue_size = 1,buff_size=2**24)
		rospy.Subscriber("/hough_lines_image/lines", LineArrayStamped, self.callback, queue_size = 1)


	def order_points(self,pts):
		xSorted = pts[np.argsort(pts[:, 0]), :]
		leftMost = xSorted[:2, :]
		rightMost = xSorted[2:, :]
		leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
		(tl, bl) = leftMost
		rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
		(tr, br) = rightMost
		#D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
		#(br, tr) = rightMost[np.argsort(D)[::-1], :]
		return np.array([tl, tr, br, bl], dtype="float32")


	def callback_cropped(self,data):
		cropped_x_y = data.data
		self.cropped_x = cropped_x_y.split(',')[0]
		self.cropped_y = cropped_x_y.split(',')[1]
		#print(cropped_x,cropped_y)

	def callback(self,data):
		self.bridge = CvBridge()
		self.shutdown_signal = False


		rospy.Subscriber("/cropped_x_y", String, self.callback_cropped, queue_size = 1,buff_size=2**24)

		rospy.Subscriber("/cropped_image", Image, self.callback_image, queue_size = 1)
		#print(len(data))
		#print("CALLBACK")
		header = data.header
		start = time.time()
		extend_lines_by_percent = 20.0
		mpompa = True
		#print('header',header)

		self.image_intersections = self.bridge.imgmsg_to_cv2(self.image_data, desired_encoding='bgr8')
		self.image_copy = self.image_intersections.copy()
		self.image_inter = Image()
		box_ratio = 2.15
		plus_minus = 0.3
		maximum_lines = 40
		lines = data.lines
		print('Received: ' + str(len(lines)) + ' lines.')
		if len(lines) <= maximum_lines:
			slopes = []
			all_data = []
			connection_pts_list = []
			pt_distances = []
			connection_pts_lines = []
			for line in lines:
				x1 = int(line.pt1.x) 
				y1 = int(line.pt1.y)
				x2 = int(line.pt2.x)
				y2 = int(line.pt2.y)

				if x1 != x2 and y1 != y2:
					if x1 < x2:
						x1_first = True
					else:
						x1_first = False
					
					slope = ((float(y2)-float(y1))/(float(x2)-float(x1)))


					if x1_first == True:
						#print('x1 first', x1,x2)
						dist_x = x2-x1
						#print('before', x1,y1,x2,y2,slope)
						diff_x = x1 - int(x1 - dist_x * extend_lines_by_percent/100)
						x1 = int(x1 - dist_x * extend_lines_by_percent/100)
						x2 = int(x2 + dist_x * extend_lines_by_percent/100)
						y1 = int(y1 - (diff_x * slope))
						y2 = int(y2  + (diff_x * slope))
						#print(dist_x,slope)
						slope_after = ((float(y2)-float(y1))/(float(x2)-float(x1)))
						

						
	#GRAMMES
						self.image_intersections = cv2.line(self.image_intersections, (x1,y1), (x2,y2), (255,0,0), 2)
						


						#print('after', x1,y1,x2,y2,slope_after)
						slopes.append(slope_after)
						all_data.append([slope_after,x1,y1,x2,y2])
					else:
						print('x2 first', x1,y1,x2,y2)		
			#print(len(slopes))			
			for e,f in itertools.combinations(slopes,2):
				line_a = slopes.index(e)
				line_b = slopes.index(f)

				if (1+f*e) != 0:
					theta = math.degrees(math.atan((f-e)/(1+f*e)))
					if theta < 0:
						theta =  - theta# + 180
					#thetas.append(theta)

					linea_pt1 = (all_data[line_a][1], all_data[line_a][2])
					linea_pt2 = (all_data[line_a][3], all_data[line_a][4])
					lineb_pt1 = (all_data[line_b][1], all_data[line_b][2])
					lineb_pt2 = (all_data[line_b][3], all_data[line_b][4])


					line1 = LineString([linea_pt1, linea_pt2])
					line2 = LineString([lineb_pt1, lineb_pt2])
					#print("DATA", all_data[line_a][3]-all_data[line_a][1],all_data[line_a][4]-all_data[line_a][2])
					try:

						connection_pt = (int(line1.intersection(line2).x),int(line1.intersection(line2).y))
						connection_pt_x = connection_pt[0]
						connection_pt_y = connection_pt[1]

						#print("slopes: ", all_data[line_a][0],all_data[line_b][0],connection_pt)

						#self.image_intersections = cv2.putText(self.image_intersections, str(theta), (connection_pt_x,connection_pt_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), cv2.LINE_AA)
						#self.image_intersections = cv2.putText(self.image_intersections, str(line_b), (connection_pt_x-10,connection_pt_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), cv2.LINE_AA)

						connection_pts_list.append(connection_pt)#,line_a,line_b,theta])
						connection_pts_lines.append([line_a,line_b])#,linea_pt1,linea_pt2,lineb_pt1,lineb_pt2])

						self.image_intersections = cv2.circle(self.image_intersections, connection_pt, 3, (0,0,255), 2)


					except:
						pass
				else:
					pass
			#print("corner_pts", len(connection_pts_list))
			#for k,l,m,n in itertools.combinations
			'''
			Making sure that I have at least 4 lines, but if I have more than 25 I have to rewadjust the hough and edge detector
			'''
			if len(connection_pts_list) >= 4 and len(connection_pts_list) <= maximum_lines:

				#print("total connection pts: ",len(connection_pts_list))
				#print(len(connection_pts_lines))
				for k,l,m,n in itertools.combinations(connection_pts_list, 4):
					four_points_list = []
					k_position = connection_pts_list.index(k)
					l_position = connection_pts_list.index(l)
					m_position = connection_pts_list.index(m)
					n_position = connection_pts_list.index(n)
					#print(k)
					

					position = k_position
					if connection_pts_lines[position][0] not in four_points_list:
						four_points_list.append(connection_pts_lines[position][0])
					if connection_pts_lines[position][1] not in four_points_list:
						four_points_list.append(connection_pts_lines[position][1])
					position = l_position
					if connection_pts_lines[position][0] not in four_points_list:
						four_points_list.append(connection_pts_lines[position][0])
					if connection_pts_lines[position][1] not in four_points_list:
						four_points_list.append(connection_pts_lines[position][1])
					position = m_position
					if connection_pts_lines[position][0] not in four_points_list:
						four_points_list.append(connection_pts_lines[position][0])
					if connection_pts_lines[position][1] not in four_points_list:
						four_points_list.append(connection_pts_lines[position][1])
					position = n_position
					if connection_pts_lines[position][0] not in four_points_list:
						four_points_list.append(connection_pts_lines[position][0])
					if connection_pts_lines[position][1] not in four_points_list:
						four_points_list.append(connection_pts_lines[position][1])
					
					if len(four_points_list) == 4:
						try:
							new_copy = []
							for element in four_points_list:
								new_copy.append(element)

							tadata = [connection_pts_lines[k_position],
								connection_pts_lines[l_position],
								connection_pts_lines[m_position],
								connection_pts_lines[n_position]]

							check_parallels = [connection_pts_lines[k_position],
								connection_pts_lines[l_position],
								connection_pts_lines[m_position],
								connection_pts_lines[n_position]]

							check_index = connection_pts_lines[k_position]
							new_copy.remove(check_index[1])
							remaining_parallels = [connection_pts_lines[k_position],
								connection_pts_lines[l_position],
								connection_pts_lines[m_position],
								connection_pts_lines[n_position]]

							check_parallels.remove(check_index)
							connecting_pt = []

							for element in check_parallels:
								if check_index[0] in element:
									if element[0] in new_copy:
										new_copy.remove(element[0])
									if element[1] in new_copy:
										new_copy.remove(element[1])

							if len(new_copy) > 0:
								#print(four_points_list,check_index[0])
								four_points_list.remove(check_index[0])
								#print(four_points_list,new_copy[0])
								four_points_list.remove(new_copy[0])
								#print("parallel lines A: ", check_index[0],new_copy[0])
								#print("parallel lines B: ", four_points_list[0],four_points_list[1])
								p1_1 = [check_index[0]]
								p1_2 = [new_copy[0]]
								p2_1 = [four_points_list[0]]
								p2_2 = [four_points_list[1]]
		#ASDASDSAFASFASDASDFASD
								for i in range(len(tadata)):
									if p1_1[0] in tadata[i]:
										p1_1.append(connection_pts_list[connection_pts_lines.index(tadata[i])])
									if p1_2[0] in tadata[i]:
										p1_2.append(connection_pts_list[connection_pts_lines.index(tadata[i])])
									if p2_1[0] in tadata[i]:
										p2_1.append(connection_pts_list[connection_pts_lines.index(tadata[i])])
									if p2_2[0] in tadata[i]:
										p2_2.append(connection_pts_list[connection_pts_lines.index(tadata[i])])


								l1_1_length = math.sqrt((p1_1[1][0] - p1_1[2][0])**2 +  (p1_1[1][1] - p1_1[2][1])**2)
								l1_2_length = math.sqrt((p1_2[1][0] - p1_2[2][0])**2 +  (p1_2[1][1] - p1_2[2][1])**2) 
								l2_1_length = math.sqrt((p2_1[1][0] - p2_1[2][0])**2 +  (p2_1[1][1] - p2_1[2][1])**2)
								l2_2_length = math.sqrt((p2_2[1][0] - p2_2[2][0])**2 +  (p2_2[1][1] - p2_2[2][1])**2) 

								if (((l1_1_length >= (l1_2_length - l1_2_length * 0.5)) and (l1_1_length <= (l1_2_length + l1_2_length * 0.5))) and 
								((l2_1_length >= (l2_2_length - l2_2_length * 0.5)) and (l2_1_length <= (l2_2_length + l2_2_length * 0.5)))):
									l1_length_average = (l1_1_length + l1_2_length)/2
									l2_length_average = (l2_1_length + l2_2_length)/2

									l_max = max(l1_length_average,l2_length_average)
									l_min = min(l1_length_average,l2_length_average)

									ratio = l_max/l_min

									#print((box_ratio - box_ratio*plus_minus))
									#print((box_ratio + box_ratio*plus_minus))
									#print(ratio)
									if ratio >= (box_ratio - box_ratio*plus_minus) and ratio <= (box_ratio + box_ratio*plus_minus) and l_min >= 60:

										self.image_intersections = cv2.circle(self.image_intersections, k, 5, (255,255,255), 1)
										self.image_intersections = cv2.circle(self.image_intersections, l, 5, (255,255,255), 1)
										self.image_intersections = cv2.circle(self.image_intersections, m, 5, (255,255,255), 1)
										self.image_intersections = cv2.circle(self.image_intersections, n, 5, (255,255,255), 1)

							
										final_list = [p1_1,p1_2,p2_1,p2_2]
										#print('final list', final_list)


										slope_1_1 = (float(p1_1[2][1])-float(p1_1[1][1]))/(float(p1_1[2][0])-float(p1_1[1][0]))
										slope_1_2 = (float(p1_2[2][1])-float(p1_2[1][1]))/(float(p1_2[2][0])-float(p1_2[1][0]))
										slope_2_1 = (float(p2_1[2][1])-float(p2_1[1][1]))/(float(p2_1[2][0])-float(p2_1[1][0]))
										slope_2_2 = (float(p2_2[2][1])-float(p2_2[1][1]))/(float(p2_2[2][0])-float(p2_2[1][0]))

										
										four_points = []
										all_points = [(p1_1[1][0],p1_1[1][1]),(p1_1[2][0],p1_1[2][1]),
													(p1_2[1][0],p1_2[1][1]),(p1_2[2][0],p1_2[2][1]),
													(p2_1[1][0],p2_1[1][1]),(p2_1[2][0],p2_1[2][1]),
													(p2_2[1][0],p2_2[1][1]),(p2_2[2][0],p2_2[2][1])]
										for point in all_points:
											if point not in four_points:
												four_points.append(point)


										pts = np.asarray(four_points)
										rect = self.order_points(pts)
										(tl, tr, br, bl) = rect

										self.image_intersections = cv2.putText(self.image_intersections, 'TL', (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_SIMPLEX,
															1, (0, 0, 255), 1, cv2.LINE_AA)
										self.image_intersections = cv2.putText(self.image_intersections, 'TR', (int(tr[0]), int(tr[1])), cv2.FONT_HERSHEY_SIMPLEX,
															1, (0, 0, 255), 1, cv2.LINE_AA)
										self.image_intersections = cv2.putText(self.image_intersections, 'BR', (int(br[0]), int(br[1])), cv2.FONT_HERSHEY_SIMPLEX,
															1, (0, 0, 255), 1, cv2.LINE_AA)
										self.image_intersections = cv2.putText(self.image_intersections, 'BL', (int(bl[0]), int(bl[1])), cv2.FONT_HERSHEY_SIMPLEX,
															1, (0, 0, 255), 1, cv2.LINE_AA)
										#print("passed")

										#print("passed2")
										message = LineArrayStamped()
										for i in range(4):
											message.lines.append([])
											message.lines[i] = Line()
											message.lines[i].pt1.x = final_list[i][1][0]
											message.lines[i].pt1.y = final_list[i][1][1]
											message.lines[i].pt2.x = final_list[i][2][0]
											message.lines[i].pt2.y = final_list[i][2][1]

										message.header = header
										#print(l1_1_length,l1_2_length,l2_1_length,l2_2_length)
										#connections = self.fourpoints.get_num_connections()
										#print("PUBLISHED MESSAGES",connections)
										#if self.fourpoints.get_num_connections() < 1:
										self.fourpoints.publish(str(tl[0]) + ',' +str(tl[1])+'/'+str(tr[0]) + ',' +str(tr[1]) + '/' + str(br[0]) + ',' +str(br[1])+ '/' +str(bl[0]) + ',' +str(bl[1]))

										self.pub.publish(message)
										self.shutdown_signal = True



										#print('mpike')
										#cv2.imshow("image",self.image_intersections)
										#cv2.waitKey()
									else:
										pass
									

						except:
							pass

		else:
			self.image_intersections = cv2.putText(self.image_intersections, 'Too many lines', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
															1, (0, 255, 0), 1, cv2.LINE_AA)
			#print("too many hough lines, adjust settings", len(lines))
		if self.shutdown_signal == True:
			self.image_intersections = cv2.putText(self.image_intersections, 'LOCKED', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
															2, (0, 0, 255), 2, cv2.LINE_AA)
		self.image_inter = self.bridge.cv2_to_imgmsg(self.image_intersections, encoding='bgr8')
		self.image_inter.header = header
	#self.publisher = rospy.Publisher("/corner_points", Image, queue_size = 1)
		self.publisher.publish(self.image_inter)

		print("The whole thing took %.3f sec.\n" % (time.time() - start))

		if self.shutdown_signal == True:
			#rospy.signal_shutdown('Found the four')
			rospy.sleep(4)
	def callback_image(self,data):
		self.image_data = data
		#print(data.width)
		#print(data.height)
		self.bridge = CvBridge()

#def main():

if __name__ == '__main__':
	rospy.init_node('intersecting_points', anonymous=True)
	sub = subscribe()
	rospy.spin()

   #main()
