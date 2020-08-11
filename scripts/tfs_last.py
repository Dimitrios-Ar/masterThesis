#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from pycpd import RigidRegistration
from pycpd import AffineRegistration
from matplotlib import pyplot as plt
from scipy import ndimage
import time
import glob
from scipy.spatial import Delaunay
import math
import cv2
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from datetime import datetime

import os 

ros = True
open3d_vis = False
numpy_vis = False
layers = 8
last_visualisation = False
all_cuboids = []
cut_in_4_more = True

depth = 0.52
final_vis = False


if ros == True:
    import rospy
    import ros_numpy
    from std_msgs.msg import Header,String
    from sensor_msgs.msg import PointCloud2, PointField
    import sensor_msgs.point_cloud2 as pc2


class subscribers():
    def __init__(self):
        
        
        """READING ROS POINTCLOUD, PREPARINT PUBLISHER"""
        self.publisher_container = rospy.Publisher('/ground_truth_pcd', PointCloud2, queue_size = 1)
        self.publisher = rospy.Publisher('/insider_points',PointCloud2,queue_size=1)
        self.pointcloud = rospy.Subscriber('/pointcloud', PointCloud2, self.pointcloud_data, queue_size = 1,buff_size=2**24)
        self.sub_once = rospy.Subscriber("/hough_pcd_plus", PointCloud2, self.rigid_registration, queue_size = 1,buff_size=2**24)


    """VISUALIZER FOR COHERENT REGISTRATION"""
    def visualize(self, iteration, error, X, Y, ax):
        plt.cla()
        ax.scatter(X[:, 0],  X[:, 1], color='orange', label='Target')
        ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
        #plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        #    iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
        ax.legend(loc='upper left', fontsize='x-large')
        plt.draw()
        plt.pause(0.001)

    def pointcloud_data(self,pcl):

        full_cloud_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcl)
        self.full_pcl = o3d.geometry.PointCloud()
        self.full_pcl.points = o3d.utility.Vector3dVector(full_cloud_array)


    def rigid_registration(self,pcl_data):
        
        """READ BOTH POINTCLOUDS"""
        self.ros_cloud = self.full_pcl
        pcl_header = pcl_data.header
        pointcloud_data = pcl_data
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pointcloud_data)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_array)



        dir_path = os.path.dirname(os.path.realpath(__file__))
        #print("CURRENT PATH",dir_path)

        pcd_2 = o3d.io.read_point_cloud("container_big_upper.ply")
        pcd_2_copy = o3d.io.read_point_cloud("container_big_upper.ply")
        pcd_box = o3d.io.read_point_cloud("container_big_downsampled.ply")



        upper_1, upper_2, upper_3, upper_4 = [0.23, -0.87, 0.24], [-0.13, -0.93, 0.23], [-0.13, -0.95, -0.53], [0.23,
                                                                                                                -0.91,
                                                                                                                -0.52]
        down_1, down_2, down_3, down_4 = [0.35, -1.3, 0.27], [-0.1, -1.4, 0.26], [-0.093, -1.4, -0.54], [0.34, -1.4,
                                                                                                         -0.53]

        octapoint_array =[upper_1,upper_2,upper_3,upper_4,down_1,down_2,down_3,down_4]

        octapoint_hull = o3d.geometry.PointCloud()
        octapoint_hull.points = o3d.utility.Vector3dVector(octapoint_array)

        """PLANE SEGMENTATION"""
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=5,
                                                 num_iterations=250)

        plane_model_2, inliers_2 = pcd_2.segment_plane(distance_threshold=0.01,
                                                 ransac_n=5,
                                                 num_iterations=250)

        plane_model_2_copy, inliers_2_copy = pcd_2_copy.segment_plane(distance_threshold=0.01,
                                                 ransac_n=5,
                                                 num_iterations=250)


        """SAVING INLIERS"""
        inlier_cloud = pcd.select_down_sample(inliers)
        inlier_cloud_2 = pcd_2.select_down_sample(inliers_2)
        inlier_cloud_2_copy = pcd_2_copy.select_down_sample(inliers_2_copy)

        """DOWNSAMPLING POINTCLOUDS"""
        downpcd = inlier_cloud
        downpcd_2 = inlier_cloud_2.voxel_down_sample(voxel_size=0.1)
        downpcd_2_copy = inlier_cloud_2_copy.voxel_down_sample(voxel_size=0.1)
        downpcd_box = pcd_box#.voxel_down_sample(voxel_size=0.1)

        """PREPARATION OF DATA AND POINTSET REGISTRATION"""
        downpcd_as_array = np.asarray(downpcd.points)
        downpcd_2_as_array = np.asarray(downpcd_2.points)
        downpcd_2_copy_as_array = np.asarray(downpcd_2_copy.points)
        downpcd_box_as_array = np.asarray(downpcd_box.points)
        pcd_box_as_array = np.asarray(pcd_box.points)
        octapoint_hull_as_array = np.asarray(octapoint_hull.points)

        X = downpcd_as_array
        Y = downpcd_2_as_array

        reg = RigidRegistration(**{'X': X, 'Y': Y})
        reg.register()
        #plt.close(fig)

        rotation = reg.R
        translation = reg.t
        scale = reg.s

        copied_Y = np.dot(scale,downpcd_2_copy_as_array) + translation
        copied_Y = np.dot(copied_Y,rotation)
        copied_box = np.dot(scale,downpcd_box_as_array) + translation
        copied_box = np.dot(copied_box,rotation)
        troti_box = np.dot(scale,pcd_box_as_array) + translation
        troti_box = np.dot(troti_box,rotation)
        octapoint = np.dot(scale, octapoint_hull_as_array) + translation
        octapoint = np.dot(octapoint, rotation)

        wonder_container = o3d.geometry.PointCloud()
        wonder_container.points = o3d.utility.Vector3dVector(copied_Y)
        wonder_container_2 = o3d.geometry.PointCloud()
        wonder_container_2.points = o3d.utility.Vector3dVector(reg.TY)
        woner_box = o3d.geometry.PointCloud()
        woner_box.points = o3d.utility.Vector3dVector(copied_box)
        troted_box = o3d.geometry.PointCloud()
        troted_box.points = o3d.utility.Vector3dVector(troti_box)


        octabox = o3d.geometry.PointCloud()
        octabox.points = o3d.utility.Vector3dVector(octapoint)

        original_translation_2 = wonder_container.get_center() - woner_box.get_center()

        wonder_container.translate(wonder_container_2.get_center(),relative = False)
        woner_box.translate(wonder_container_2.get_center(), relative=False)
        octabox.translate(wonder_container_2.get_center(), relative= False)

        woner_box.translate(original_translation_2,relative=True)
        woner_box.rotate([[-1,0,0],[0,-1,0],[0,0,-1]], center = True)
        octabox.translate(original_translation_2, relative=True)
        octabox.rotate([[-1,0,0],[0,-1,0],[0,0,-1]], center = True)
        #woner_box.translate([0.05,0.0,0.0], relative=True)
      
          
        """PUBLISH RESULT IN ROS"""
        
        numpy_open3d_box = np.asarray(woner_box.points)
        self.array_to_pcd = pc2.create_cloud_xyz32(pcl_header, numpy_open3d_box)
        self.publisher_container.publish(self.array_to_pcd)

        numpy_open3d = np.asarray(octabox.points)
        self.scaled_polygon_pcl = pc2.create_cloud_xyz32(pcl_header, numpy_open3d)
        self.publisher.publish(self.scaled_polygon_pcl)


if __name__ == "__main__":
    if ros == True:
        rospy.init_node('tfs_and_registration', anonymous=True)
        sub = subscribers()
        rospy.spin()
    elif ros == False:
        sub = subscribers()
	#main()