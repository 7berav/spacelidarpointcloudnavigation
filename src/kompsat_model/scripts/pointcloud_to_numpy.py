#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import numpy as np

class PointCloudSaver:
    def __init__(self):
        self.sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.callback)
        self.saved = False

    def callback(self, msg):
        if not self.saved:
            cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            point_array = np.array(cloud_points)

            # 저장 (예: npy 파일로)
            np.save("/home/smrl/cylinder_high.npy", point_array)
            rospy.loginfo("Saved pointcloud with shape: {}".format(point_array.shape))

            self.saved = True  # 한 번만 저장하도록 설정

if __name__ == '__main__':
    rospy.init_node('pointcloud_saver')
    pcs = PointCloudSaver()
    rospy.spin()
