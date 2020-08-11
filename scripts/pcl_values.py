#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import rospy
import open3d as o3d
import math
import time
import ros_numpy
import sensor_msgs.point_cloud2 as pc2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from std_msgs.msg import Header,String
from sensor_msgs.msg import PointCloud2, PointField
from skimage.measure import LineModelND, ransac
from opencv_apps.msg import LineArrayStamped
from opencv_apps.msg import Line
import sys


class subscribers():
	def __init__(self):
		self.image_width = 496
		self.image_height = 250
		self.publisher = rospy.Publisher('/hough_pcd_plus', PointCloud2, queue_size = 1)
		self.sub_once = rospy.Subscriber("/pointcloud", PointCloud2, self.pcl_callback, queue_size = 1,buff_size=2**24)
		rospy.Subscriber("/boundary_lines", LineArrayStamped, self.lines_callback, queue_size = 1, buff_size=2**24)

	def pcl_callback(self,pcl_data):
		self.pcl_height = pcl_data.height
		self.pcl_width = pcl_data.width
		self.pcl_header = pcl_data.header
		self.pointcloud_data = pcl_data
		self.pointcloud_fields = pcl_data.fields
		self.pcl_header_seq = pcl_data.header.seq
		#print(self.pointcloud_fields)

	def lines_callback(self,lines_data):
		#start = time.time()
		self.visualization = False
		four_neighborhoods = []
		all_points= []
		interest_points = []
		self.rectangle_before_ransac = np.empty((0,3), float)
		self.rectangle_after_ransac = np.empty((0,3), float)
		self.rect_before_ransac = []
		self.rect_after_ransac = []
		self.counter_2 = 0

		self.start = time.time()

		lines = lines_data.lines

		line_1 = np.array([[int(lines[0].pt1.x),int(lines[0].pt1.y)],[int(lines[0].pt2.x),int(lines[0].pt2.y)]])
		line_2 = np.array([[int(lines[1].pt1.x),int(lines[1].pt1.y)],[int(lines[1].pt2.x),int(lines[1].pt2.y)]])
		line_3 = np.array([[int(lines[2].pt1.x),int(lines[2].pt1.y)],[int(lines[2].pt2.x),int(lines[2].pt2.y)]])
		line_4 = np.array([[int(lines[3].pt1.x),int(lines[3].pt1.y)],[int(lines[3].pt2.x),int(lines[3].pt2.y)]])

		self.xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.pointcloud_data)
		
		#fig = plt.figure()

		interest_points.append(self.connect2(line_1))
		self.prepare_points(interest_points)
   		interest_points = []

		interest_points.append(self.connect2(line_2))
		self.prepare_points(interest_points)
   		interest_points = []

		interest_points.append(self.connect2(line_3))
		self.prepare_points(interest_points)
   		interest_points = []

		interest_points.append(self.connect2(line_4))
		self.prepare_points(interest_points)
   		interest_points = []

		#ax= fig.add_subplot(2,1,1, projection='3d')
		#ax.scatter(self.rectangle_after_ransac[:, 0], self.rectangle_after_ransac[:, 1], self.rectangle_after_ransac[:, 2], c='b',
        #  marker='o')#, label='Inlier data')
		#ax = fig.add_subplot(2,1,2, projection='3d')
		#ax.scatter(self.rectangle_before_ransac[:, 0], self.rectangle_before_ransac[:, 1], self.rectangle_before_ransac[:, 2], c='r',
        #  marker='x')
		'''
    
		if self.visualization == True:
		#	self.sub_once.unregister()
			pcd = o3d.geometry.PointCloud()
    		pcd.points = o3d.utility.Vector3dVector(self.rectangle_before_ransac)
    		selected_pcd = o3d.geometry.PointCloud()
    		selected_pcd.points = o3d.utility.Vector3dVector(self.rectangle_after_ransac)
    		#cl, ind = selected_pcd.remove_statistical_outlier(nb_neighbors=10,
             #                                           std_ratio=2.0)
    		#inlier_cloud = selected_pcd.select_down_sample(ind)
    		#outlier_cloud = selected_pcd.select_down_sample(ind, invert=True)

    		pcd.paint_uniform_color([1, 0.706, 0])
    		#outlier_cloud.paint_uniform_color([0, 0.651, 0.929])
    		selected_pcd.paint_uniform_color([0, 0.651, 0.929])
    		#print(sys.path)
    		name = "test_saves/pcd_" + str(self.pcl_header_seq) + ".ply"
    		o3d.io.write_point_cloud(name, selected_pcd)
		
		'''


		selected_pcd = o3d.geometry.PointCloud()
		selected_pcd.points = o3d.utility.Vector3dVector(self.rectangle_after_ransac)
		numpy_open3d = np.asarray(selected_pcd.points)
		before_ransac_pcd = o3d.geometry.PointCloud()
		before_ransac_pcd.points = o3d.utility.Vector3dVector(self.rectangle_before_ransac)

		before_ransac_pcd.paint_uniform_color([1, 0.706, 0])
		selected_pcd.paint_uniform_color([0, 0.651, 0.929])
		#o3d.visualization.draw_geometries([before_ransac_pcd,selected_pcd])

		self.scaled_polygon_pcl = pc2.create_cloud_xyz32(self.pcl_header, numpy_open3d)
		self.publisher.publish(self.scaled_polygon_pcl)

		print("The whole thing took %.3f sec.\n" % (time.time() - self.start))
		'''
		pcl_points = pc2.read_points(self.pointcloud_data, skip_nans=True)
		self.counter = 0
		for point in pcl_points:
			pt_x = point[0]
			pt_y = point[1]
			pt_z = point[2]
			all_points.append((pt_x,pt_y,pt_z))
			self.counter += 1
		'''



	def prepare_points(self,interest_pts):
		self.corresponding_points = []
		all_neighborhood_pts = []
		for neighborhood in interest_pts:
			for pt in neighborhood:
				neighborhood_points = [(pt[0],pt[1]), (pt[0] + 1, pt[1] + 1),
				(pt[0] - 1, pt[1]), (pt[0] - 1, pt[1] - 1), (pt[0], pt[1] - 1),
				(pt[0] + 1, pt[1]), (pt[0], pt[1] + 1), (pt[0] - 1, pt[1] + 1),
				(pt[0] + 1, pt[1] - 1)]

				for point in neighborhood_points:
					pcl_point = self.image_width * point[1] + point[0]
					if pcl_point not in self.corresponding_points:
						self.corresponding_points.append(pcl_point)

		for tada in self.corresponding_points:
			pt_x = self.xyz_array[tada][0]
			pt_y = self.xyz_array[tada][1]
			pt_z = self.xyz_array[tada][2]
			if pt_x != 0.0:
				all_neighborhood_pts.append((pt_x,pt_y,pt_z))
				self.counter_2 += 1

		self.run_ransac(all_neighborhood_pts)


	def run_ransac(self,pts):
		xyz = np.array(pts)
		model_robust, inliers = ransac(xyz, LineModelND, min_samples=10,
                               residual_threshold=0.01, max_trials=100)
		outliers = inliers == False
		self.rectangle_after_ransac = np.append(self.rectangle_after_ransac, xyz[inliers], axis = 0)
		self.rectangle_before_ransac = np.append(self.rectangle_before_ransac, xyz, axis = 0)

	def connect2(self,ends):
	    d0, d1 = np.diff(ends, axis=0)[0]
	    if np.abs(d0) > np.abs(d1):
	        return np.c_[np.arange(ends[0, 0], ends[1,0] + np.sign(d0), np.sign(d0), dtype=np.int32),
	                     np.arange(ends[0, 1] * np.abs(d0) + np.abs(d0)//2,
	                               ends[0, 1] * np.abs(d0) + np.abs(d0)//2 + (np.abs(d0)+1) * d1, d1, dtype=np.int32) // np.abs(d0)]
	    else:
	        return np.c_[np.arange(ends[0, 0] * np.abs(d1) + np.abs(d1)//2,
	                               ends[0, 0] * np.abs(d1) + np.abs(d1)//2 + (np.abs(d1)+1) * d0, d0, dtype=np.int32) // np.abs(d1),
	                     np.arange(ends[0, 1], ends[1,1] + np.sign(d1), np.sign(d1), dtype=np.int32)]


if __name__ == "__main__":
	rospy.init_node('pointcloud_4_points', anonymous=True)
	sub = subscribers()
	rospy.spin()
