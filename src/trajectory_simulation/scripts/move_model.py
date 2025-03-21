#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import math
import tf
from tf.transformations import quaternion_from_euler
import numpy as np

def move_model():
    rospy.init_node('gazebo_model_mover', anonymous=True)
    rospy.wait_for_service('/gazebo/set_model_state')  # Gazebo 서비스 대기
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)  # 서비스 연결

    rate = rospy.Rate(10)  # 10Hz (0.1초마다 위치 업데이트)
    t = -2000  # 시간 변수

    while not rospy.is_shutdown():
        # 원하는 Trajectory 방정식 (CW equation)
        n = 1.07 * 10 ** (-3)

        if t <= 0:
            C1, C2, C3, C4 = -30*(n/2), 0, 10, -20
            q = quaternion_from_euler(0, np.radians(90), 0)
        else:
            C1, C2, C3, C4 = 0, -30, 0, 0

        x = (2/n)*C1 + C2*math.cos(n*t) + C3*math.sin(n*t)
        y = -3*C1*t -2*C2*math.sin(n*t) + 2*C3*math.cos(n*t) + C4
        z = 0  # 고정된 높이

        if t >= 0:
            yaw = math.atan2(-y, -x)
            q = quaternion_from_euler(0, np.radians(90), yaw)

        # Gazebo 모델 상태 설정
        model_state = ModelState()
        model_state.model_name = "KOMPSAT"  # Gazebo의 모델 이름
        model_state.pose.position.x = x
        model_state.pose.position.y = y
        model_state.pose.position.z = z
        model_state.pose.orientation.x = q[0]
        model_state.pose.orientation.y = q[1]
        model_state.pose.orientation.z = q[2]
        model_state.pose.orientation.w = q[3]

        # Gazebo 서비스 호출하여 모델 위치 업데이트
        try:
            set_state(model_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

        t += 10  # 시간 업데이트
        rate.sleep()

if __name__ == '__main__':
    try:
        move_model()
    except rospy.ROSInterruptException:
        pass