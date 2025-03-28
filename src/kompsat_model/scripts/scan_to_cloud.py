#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
from sensor_msgs.msg import MultiEchoLaserScan, PointCloud2
from sensor_msgs import point_cloud2

class MultiEchoLaserScanToPointCloud:
    def __init__(self):
        rospy.init_node("multi_echo_scan_to_cloud")
        
        # MultiEchoLaserScan 메시지를 받는 Subscriber 생성
        self.scan_sub = rospy.Subscriber("/lidar/scan", MultiEchoLaserScan, self.scan_callback)
        
        # PointCloud2 메시지를 퍼블리시할 Publisher 생성
        self.pc_pub = rospy.Publisher("/lidar/points", PointCloud2, queue_size=10)

    def scan_callback(self, scan_msg):
        points = []
        
        # 각 레이저 에코에 대한 수평, 수직 각도 및 거리 계산
        for i, dist in enumerate(scan_msg.ranges):
            if dist == float('Inf') or dist < scan_msg.range_min or dist > scan_msg.range_max:
                continue
            
            # 수평 각도 계산
            angle = scan_msg.angle_min + i * scan_msg.angle_increment

            # 수직 각도 (tilt) 계산 (수직 레이저의 각도를 사용)
            # 예: 수직 레이저가 16개라면 0 ~ 1.57 라디안으로 분포한다고 가정
            tilt = math.radians(i % 16 * 10)  # 예시로 수직 레이저를 나누어서 계산

            # x, y, z는 수평 각도, 수직 각도 및 거리 정보를 사용하여 계산
            x = dist * math.cos(angle) * math.cos(tilt)
            y = dist * math.sin(angle) * math.cos(tilt)
            z = dist * math.sin(tilt)

            points.append([x, y, z])

        # PointCloud2 생성 및 퍼블리시
        cloud = point_cloud2.create_cloud_xyz32(scan_msg.header, points)
        self.pc_pub.publish(cloud)

if __name__ == "__main__":
    node = MultiEchoLaserScanToPointCloud()
    rospy.spin()

