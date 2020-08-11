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
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime

import os 
from sensor_msgs.msg import Image,CameraInfo

ros = True
open3d_vis = False
numpy_vis = False
layers = 8
last_visualisation = False
all_cuboids = []
cut_in_4_more = True

depth = 0.52
final_vis = False

volume_estimation = True

if ros == True:
    import rospy
    import ros_numpy
    from std_msgs.msg import Header,String
    from sensor_msgs.msg import PointCloud2, PointField
    import sensor_msgs.point_cloud2 as pc2


class subscribers():
    def __init__(self):
        #self.start = datetime.now()

        # Statements


        self.ros = ros
        """CREATING VISUALIZER FOR TESTS"""
    
        """READING ROS POINTCLOUD, PREPARINT PUBLISHER"""
        #self.publisher_container = rospy.Publisher('/ground_truth_pcd', PointCloud2, queue_size = 1)
        self.pub = rospy.Publisher('/volume_estimation',Image, queue_size=1)
        self.pointcloud = rospy.Subscriber('/pointcloud', PointCloud2, self.pointcloud_data, queue_size = 1,buff_size=2**24)
        self.octabox = rospy.Subscriber("/insider_points", PointCloud2, self.octabox, queue_size = 1,buff_size=2**24)





    def make_plots(self,upper_1,upper_2,upper_3,upper_4,down_1,down_2,down_3,down_4):

        octapoint_array = [upper_1, upper_2, upper_3, upper_4, down_1, down_2, down_3, down_4]
        #print(octapoint_array)

        subdivisions_long = 2
        subdivisions_short = 2
        i = 1
        j = 1


        #f#or i in range(1,subdivisions_long):
        #   for j in range(1,subdivisions_short):
        upper_middle_long_14 = [(upper_1[0] + upper_4[0]) * i/ subdivisions_long,
                             (upper_1[1] + upper_4[1]) *i/ subdivisions_long,
                             (upper_1[2] + upper_4[2]) *i/ subdivisions_long]
        upper_middle_short_12 = [(upper_1[0] + upper_2[0]) * j/ subdivisions_short,
                              (upper_1[1] + upper_2[1]) *j/ subdivisions_short,
                              (upper_1[2] + upper_2[2]) *j/ subdivisions_short]
        down_middle_long_14 = [(down_1[0] + down_4[0]) * i / subdivisions_long,
                             (down_1[1] + down_4[1]) * i / subdivisions_long,
                             (down_1[2] + down_4[2]) * i / subdivisions_long]
        down_middle_short_12 = [(down_1[0] + down_2[0]) * j / subdivisions_short,
                              (down_1[1] + down_2[1]) * j / subdivisions_short,
                              (down_1[2] + down_2[2]) * j / subdivisions_short]

        upper_middle_long_23 = [(upper_2[0] + upper_3[0]) * i / subdivisions_long,
                             (upper_2[1] + upper_3[1]) * i / subdivisions_long,
                             (upper_2[2] + upper_3[2]) * i / subdivisions_long]
        upper_middle_short_43 = [(upper_4[0] + upper_3[0]) * j / subdivisions_short,
                              (upper_4[1] + upper_3[1]) * j / subdivisions_short,
                              (upper_4[2] + upper_3[2]) * j / subdivisions_short]
        down_middle_long_23 = [(down_2[0] + down_3[0]) * i / subdivisions_long,
                            (down_2[1] + down_3[1]) * i / subdivisions_long,
                            (down_2[2] + down_3[2]) * i / subdivisions_long]
        down_middle_short_43 = [(down_4[0] + down_3[0]) * j / subdivisions_short,
                             (down_4[1] + down_3[1]) * j / subdivisions_short,
                             (down_4[2] + down_3[2]) * j / subdivisions_short]

        upper_middle = [(upper_middle_long_14[0] + upper_middle_long_23[0]) * i / subdivisions_long,
                             (upper_middle_long_14[1] + upper_middle_long_23[1]) * i / subdivisions_long,
                             (upper_middle_long_14[2] + upper_middle_long_23[2]) * i / subdivisions_long]
        down_middle = [(down_middle_long_14[0] + down_middle_long_23[0]) * i / subdivisions_long,
                             (down_middle_long_14[1] + down_middle_long_23[1]) * i / subdivisions_long,
                             (down_middle_long_14[2] + down_middle_long_23[2]) * i / subdivisions_long]


        plots = [
            [upper_2, upper_middle_short_12, upper_middle, upper_middle_long_23, down_2, down_middle_short_12, down_middle,
             down_middle_long_23],
            [upper_1, upper_middle_short_12, upper_middle, upper_middle_long_14, down_1, down_middle_short_12, down_middle,
             down_middle_long_14],
            [upper_3, upper_middle_short_43, upper_middle, upper_middle_long_23, down_3, down_middle_short_43, down_middle,
             down_middle_long_23],
            [upper_4, upper_middle_short_43, upper_middle, upper_middle_long_14, down_4, down_middle_short_43, down_middle,
             down_middle_long_14]
            ]
        
        return plots



    def distance_calculation(self,cuboid):

        upper_boundary = [cuboid[0], cuboid[1], cuboid[2], cuboid[3]]
        upper_boundary_cloud = o3d.geometry.PointCloud()
        upper_boundary_cloud.points = o3d.utility.Vector3dVector(upper_boundary)
        self.bounding_box = upper_boundary_cloud.get_oriented_bounding_box()

        p1 = upper_boundary[0]
        p2 = upper_boundary[1]
        p3 = upper_boundary[2]
        shortest_pt = p1

        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1

        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp

        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, p3)


        self.in_hull(np.asarray(self.ros_cloud.points), cuboid)
        cl, ind = self.pcd_insider.remove_statistical_outlier(nb_neighbors=5, std_ratio=2.0)
        inlier_cloud = self.pcd_insider.select_down_sample(ind)
        #print(datetime.now() - self.start)
        # print("shortest point", shortest_pt)
        '''FINDING SHORTEST DISTANCE OF EACH POINT TO UPPER BOUNDARY'''
        shortest_distance_calc = self.shortest_distance(shortest_pt, np.asarray(inlier_cloud.points), a, b, c, d)

        '''

                '''

        return shortest_distance_calc

    def in_hull(self, p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """

        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        counter = 0
        insider_points = []
        result = (hull.find_simplex(p) >= 0)
        for i in range(len(result)):
            if result[i] == True:
                counter += 1
                insider_points.append(p[i])
        #print("points_in_box", counter)
        self.pcd_insider = o3d.geometry.PointCloud()
        self.pcd_insider.points = o3d.utility.Vector3dVector(np.asarray(insider_points))


    def pointcloud_data(self,pcl):

        full_cloud_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcl)
        self.full_pcl = o3d.geometry.PointCloud()
        self.full_pcl.points = o3d.utility.Vector3dVector(full_cloud_array)

    # Function to find distance
    def shortest_distance(self, shortest_pt, cloud_list, alfa, bita, ci, di):
        a = alfa
        b = bita
        c = ci
        d = di
        d = 0
        e = 0
        xi, yi, zed = shortest_pt[0], shortest_pt[1], shortest_pt[2]
        d = abs((a * xi + b * yi + c * zed + d))
        e = (math.sqrt(a * a + b * b + c * c))
        distance = d / e
        all_distances, all_distances_plus = [],[]
        shortest_distance = d/e
        for point in cloud_list:
            d = 0
            e = 0
            xi,yi,zed = point[0],point[1],point[2]
            d = abs((a * xi + b * yi + c * zed + d))
            e = (math.sqrt(a * a + b * b + c * c))
            distance = d/e
            real_distance = distance-shortest_distance
            decimal_distance = "{:.3f}".format(real_distance)
            all_distances_plus.append([xi,yi,zed,real_distance])
            all_distances.append(real_distance)


        #print(shortest_distance)
        #print(all_distances)
        step = abs(max(all_distances) - min(all_distances)) / layers
        #print(step)
        #print("min",min(all_distances))
        #print("max",max(all_distances))
        #print(all_distances_plus)
        all_distances_plus.sort(key=lambda x: x[3])
        #print(all_distances_plus)
        counter_1,counter_2,counter_3,counter_4,counter_5 = 0,0,0,0,0
        cloud_array_1,cloud_array_2, cloud_array_3, cloud_array_4,cloud_array_5 = [],[],[],[],[]

        for i in all_distances_plus:
            if i[3] <= step:
                counter_1 += 1
                cloud_array_1.append([i[0],i[1],i[2]])
            elif i[3] <= (step * 2):
                counter_2 += 1
                cloud_array_2.append([i[0],i[1],i[2]])
            elif i[3] <= (step * 3):
                counter_3 += 1
                cloud_array_3.append([i[0], i[1], i[2]])
            elif i[3] <= (step * 4):
                counter_4 += 1
                cloud_array_4.append([i[0], i[1], i[2]])
            else:
                counter_5 += 1
                cloud_array_5.append([i[0],i[1],i[2]])

        cloud_1 = o3d.geometry.PointCloud()
        cloud_2 = o3d.geometry.PointCloud()
        cloud_3 = o3d.geometry.PointCloud()
        cloud_4 = o3d.geometry.PointCloud()
        cloud_5 = o3d.geometry.PointCloud()

        cloud_1.points = o3d.utility.Vector3dVector(cloud_array_1)
        cloud_2.points = o3d.utility.Vector3dVector(cloud_array_2)
        cloud_3.points = o3d.utility.Vector3dVector(cloud_array_3)
        cloud_4.points = o3d.utility.Vector3dVector(cloud_array_4)
        cloud_5.points = o3d.utility.Vector3dVector(cloud_array_5)

        cloud_1.paint_uniform_color([0.2,0,0.3])
        cloud_2.paint_uniform_color([0.2,0,0.8])
        cloud_3.paint_uniform_color([0.2,0.5,0.3])
        cloud_4.paint_uniform_color([0.2,0.8,1])
        cloud_5.paint_uniform_color([0.2,1,0.8])

        #print("all_distances_are", all_distances)
        all_distances.sort()
        #print("sorted_distances", all_distances)
        #print(len(all_distances))
        #print(np.median(all_distances))
        #print(np.mean(all_distances))
        #print()
        if last_visualisation == True:
            o3d.visualization.draw_geometries([cloud_1, cloud_2, cloud_3, cloud_4, cloud_5])

        return np.median(all_distances)


    def octabox(self,pcl_data):
      try:
        bridge = CvBridge()
        self.header = pcl_data.header
        if self.ros == True:
            """READ BOTH POINTCLOUDS"""
            self.ros_cloud = self.full_pcl
            pcl_header = pcl_data.header
            octabox = pcl_data
            #xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pointcloud_data)
            #pcd = o3d.geometry.PointCloud()
            #pcd.points = o3d.utility.Vector3dVector(xyz_array)



        if volume_estimation == True:
          make_plots_points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(octabox)
          total_volume = self.distance_calculation(np.asarray(make_plots_points))
          print("totalVOLUME",total_volume)
          
          
          grids = self.make_plots(make_plots_points[0],make_plots_points[1],make_plots_points[2],make_plots_points[3],make_plots_points[4],make_plots_points[5],make_plots_points[6],make_plots_points[7])

          #print(len(grids))
          #print(grids)

          for x in range(4):

              sub_grids = self.make_plots(grids[x][0],grids[x][1],grids[x][2],grids[x][3],grids[x][4],grids[x][5],grids[x][6],grids[x][7])
      
              for i in range(len(sub_grids)):
                  all_cuboids.append(sub_grids[i])


          grid_1 = o3d.geometry.PointCloud()
          grid_1.points = o3d.utility.Vector3dVector(all_cuboids[0])
          grid_2 = o3d.geometry.PointCloud()
          grid_2.points = o3d.utility.Vector3dVector(all_cuboids[1])
          grid_3 = o3d.geometry.PointCloud()
          grid_3.points = o3d.utility.Vector3dVector(all_cuboids[2])
          grid_4 = o3d.geometry.PointCloud()
          grid_4.points = o3d.utility.Vector3dVector(all_cuboids[3])
          grid_5 = o3d.geometry.PointCloud()
          grid_5.points = o3d.utility.Vector3dVector(all_cuboids[4])
          grid_6 = o3d.geometry.PointCloud()
          grid_6.points = o3d.utility.Vector3dVector(all_cuboids[5])
          grid_7 = o3d.geometry.PointCloud()
          grid_7.points = o3d.utility.Vector3dVector(all_cuboids[6])
          grid_8 = o3d.geometry.PointCloud()
          grid_8.points = o3d.utility.Vector3dVector(all_cuboids[7])
          grid_9 = o3d.geometry.PointCloud()
          grid_9.points = o3d.utility.Vector3dVector(all_cuboids[8])
          grid_10 = o3d.geometry.PointCloud()
          grid_10.points = o3d.utility.Vector3dVector(all_cuboids[9])
          grid_11 = o3d.geometry.PointCloud()
          grid_11.points = o3d.utility.Vector3dVector(all_cuboids[10])
          grid_12 = o3d.geometry.PointCloud()
          grid_12.points = o3d.utility.Vector3dVector(all_cuboids[11])
          grid_13 = o3d.geometry.PointCloud()
          grid_13.points = o3d.utility.Vector3dVector(all_cuboids[12])
          grid_14 = o3d.geometry.PointCloud()
          grid_14.points = o3d.utility.Vector3dVector(all_cuboids[13])
          grid_15 = o3d.geometry.PointCloud()
          grid_15.points = o3d.utility.Vector3dVector(all_cuboids[14])
          grid_16 = o3d.geometry.PointCloud()
          grid_16.points = o3d.utility.Vector3dVector(all_cuboids[15])


          #print(sub_grids[0])
          #print(all_cuboids[0])

          #o3d.visualization.draw_geometries([self.ros_cloud, grid_1,grid_2,grid_3,grid_4,grid_5,grid_6,grid_7,grid_8,grid_9,grid_10,grid_11,grid_12,grid_13,grid_14,grid_15,grid_16])
          #octabox.paint_uniform_color([0.5,0.5,0.7])
          #o3d.visualization.draw_geometries([grid_1,grid_2,grid_3,grid_4,grid_5,grid_6,grid_7,grid_8,grid_9,grid_10,grid_11,grid_12,grid_13,grid_14,grid_15,grid_16,octabox])
          total_volume = self.distance_calculation(np.asarray(make_plots_points))
          print("totalVOLUME",total_volume)

          self.start = datetime.now()
          median_grid_1 = self.distance_calculation(np.asarray(all_cuboids[0]))
          median_grid_2 = self.distance_calculation(np.asarray(all_cuboids[1]))
          median_grid_3 = self.distance_calculation(np.asarray(all_cuboids[2]))
          median_grid_4 = self.distance_calculation(np.asarray(all_cuboids[3]))
          median_grid_5 = self.distance_calculation(np.asarray(all_cuboids[4]))
          median_grid_6 = self.distance_calculation(np.asarray(all_cuboids[5]))
          median_grid_7 = self.distance_calculation(np.asarray(all_cuboids[6]))
          median_grid_8 = self.distance_calculation(np.asarray(all_cuboids[7]))
          median_grid_9 = self.distance_calculation(np.asarray(all_cuboids[8]))
          median_grid_10 = self.distance_calculation(np.asarray(all_cuboids[9]))
          median_grid_11 = self.distance_calculation(np.asarray(all_cuboids[10]))
          median_grid_12 = self.distance_calculation(np.asarray(all_cuboids[11]))
          median_grid_13 = self.distance_calculation(np.asarray(all_cuboids[12]))
          median_grid_14 = self.distance_calculation(np.asarray(all_cuboids[13]))
          median_grid_15 = self.distance_calculation(np.asarray(all_cuboids[14]))
          median_grid_16 = self.distance_calculation(np.asarray(all_cuboids[15]))

          #fig = plt.figure()

          #plt.clf()

          #ax = fig.gca(projection='3d')




      # plt.show()
          all_centers = []
          for i in range(len(all_cuboids)):
              sum_x,sum_y,sum_z = 0,0,0
              for j in range(len(all_cuboids[i])):
                  sum_x += all_cuboids[i][j][0]
                  sum_y += all_cuboids[i][j][1]
                  sum_z += all_cuboids[i][j][2]


          Z = [[median_grid_14,median_grid_16,median_grid_8,median_grid_6],
               [median_grid_13,median_grid_15,median_grid_7,median_grid_5],
               [median_grid_9,median_grid_11,median_grid_3,median_grid_1],
               [median_grid_10,median_grid_12,median_grid_4,median_grid_2]]

          Z = [[median_grid_2, median_grid_4, median_grid_12, median_grid_10],
               [median_grid_1, median_grid_3, median_grid_11, median_grid_9],
               [median_grid_5, median_grid_7, median_grid_15, median_grid_13],
               [median_grid_6, median_grid_8, median_grid_16, median_grid_14]]


          x = np.arange(0, 10, 2)  # len = 11
          y = np.arange(0, 5, 1)  # len = 7
          #print(Z[0])
          #print(Z,x,y)

          fig, ax = plt.subplots()
          #ax.pcolormesh(x, y, Z, cmap='RdBu', vmin=-z_max, vmax=z_max)
          im = ax.pcolormesh(x, y, Z,cmap="GnBu", vmin=0.0, vmax=0.5)
          fig.colorbar(im, ax=ax)
          plt.axis("off")
          plt.savefig('temp.png')
          temp = cv2.imread('temp.png')

          height = 300
          width = temp.shape[1] # keep original height
          dim = (width, height)

          # resize image
          resized = cv2.resize(temp, dim, interpolation = cv2.INTER_AREA)
          #cv2.imshow("temp",resized)
          #cv2.waitKey()

          self.volume_image = bridge.cv2_to_imgmsg(resized, encoding='bgr8')
          self.volume_image.header = self.header
          self.pub.publish(self.volume_image)
        
      except:
        pass



if __name__ == "__main__":
    if ros == True:
        rospy.init_node('tfs_and_registration', anonymous=True)
        sub = subscribers()
        rospy.spin()
    elif ros == False:
        sub = subscribers()
	#main()