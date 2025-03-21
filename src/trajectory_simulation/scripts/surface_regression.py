#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sympy import symbols, expand, lambdify, diff, simplify
from itertools import combinations_with_replacement
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import rospy
import math
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import trimesh

def generate_random_points_on_hexagon_prism(N):
    points = []

    # 윗면 & 아랫면
    while len(points) < N:
        x = np.random.rand()
        y = np.random.rand()
        z2 = 1

        if z2 > 0:
            z = 1
            z_values = [z, -z]

            # store valid (x, y, z) combinations
            for z_val in z_values:
                if len(points) < N:
                    points.append([x-y*0.5, y*math.sqrt(3)/2, z_val])
                else:
                    break
                if len(points) < N:
                    points.append([x-y*0.5, -y*math.sqrt(3)/2, z_val])
                else:
                    break
                if len(points) < N:
                    points.append([-x*0.5-y*0.5, x*math.sqrt(3)/2-y*math.sqrt(3)/2, z_val])
                else:
                    break

        # 앞뒤면

        x = np.random.rand() - 0.5
        z = 2 * np.random.rand() - 1

        y2 = 3/4

        if y2 >= 0:
            # calculate real x values
            y = math.sqrt(y2)
            y_values = [y, -y]

            # store valid (x, y, z) combinations
            for y_val in y_values:
                if len(points) < N:
                    points.append([x, y_val, z])
                else:
                    break
        
        #옆면
        x = np.random.rand() - 0.5
        z = 2 * np.random.rand() - 1

        y2 = 3/4

        if y2 >= 0:
            # calculate real x values
            y = math.sqrt(y2)
            y_values = [y, -y]

            # store valid (x, y, z) combinations
            for y_val in y_values:
                if len(points) < N:
                    points.append([x*0.5 - y_val*math.sqrt(3)/2, x*math.sqrt(3)/2+y_val*0.5,z])
                else:
                    break
        
        x = np.random.rand() - 0.5
        z = 2 * np.random.rand() - 1

        y2 = 3/4

        if y2 >= 0:
            # calculate real x values
            y = math.sqrt(y2)
            y_values = [y, -y]

            # store valid (x, y, z) combinations
            for y_val in y_values:
                if len(points) < N:
                    points.append([x*0.5 + y_val*math.sqrt(3)/2, -x*math.sqrt(3)/2+y_val*0.5,z])
                else:
                    break
    
    return np.array(points[:N])

def generate_random_points_on_cube(N):
    points = []  # 빈 리스트로 초기화
    while len(points) < N:
        x = 2 * np.random.rand() - 1
        y = 2 * np.random.rand() - 1
        z = 1
        z_values = [z, -z]
        
        # 유효한 (x, y, z) 조합을 저장
        for z_val in z_values:
            if len(points) < N:
                points.append([x, y, z_val])  # 새로운 점 추가
            else:
                break
        
        y = 2 * np.random.rand() - 1
        z = 2 * np.random.rand() - 1
        
        x1 = 1
        
        if x >= 0:
            # 실제 x 값을 계산
            x = x1
            x_values = [x, -x]
            
            # 유효한 (x, y, z) 조합을 저장
            for x_val in x_values:
                if len(points) < N:
                    points.append([x_val, y, z])  # 새로운 점 추가
                else:
                    break
        
        z = 2 * np.random.rand() - 1
        x = 2 * np.random.rand() - 1
        
        y1 = 1
        
        if y1 >= 0:
            # 실제 y 값을 계산
            y = y1
            y_values = [y, -y]
            
            # 유효한 (x, y, z) 조합을 저장
            for y_val in y_values:
                if len(points) < N:
                    points.append([x, y_val, z])  # 새로운 점 추가
                else:
                    break
    
    return np.array(points[:N])  # 결과를 numpy 배열로 반환

def generate_random_points_on_cylinder(N):
    points = []
    while len(points) < N:
        x = 2 * np.random.rand() - 1
        y = 2 * np.random.rand() - 1

        z2 = 1 - x**2 - y**2

        if z2 >= 0:
            # calculate real z values
            z = 1
            z_values = [z, -z]

            # store valid (x, y, z) combinations
            for z_val in z_values:
                if len(points) < N:
                    points.append([x, y, z_val])
                else:
                    break

        y = 2 * np.random.rand() - 1
        z = 2 * np.random.rand() - 1

        x2 = 1 - y**2

        if x2 >= 0:
            # calculate real x values
            x = math.sqrt(x2)
            x_values = [x, -x]

            # store valid (x, y, z) combinations
            for x_val in x_values:
                if len(points) < N:
                    points.append([x_val, y, z])
                else:
                    break
        
        z = 2 * np.random.rand() - 1
        x = 2 * np.random.rand() - 1

        y2 = 1 - x**2

        if y2 >= 0:
            # calculate real z values
            y = math.sqrt(y2)
            y_values = [y, -y]

            # store valid (x, y, z) combinations
            for y_val in y_values:
                if len(points) < N:
                    points.append([x, y_val, z])
                else:
                    break

    return np.array(points[:N])

def generate_random_points_on_surface(N):
    points = []
    while len(points) < N:
        x = 2 * np.random.rand() - 1
        y = 2 * np.random.rand() - 1

        z4 = 1 - x**4 - y**4

        if z4 >= 0:
            z = z4**(1/4)
            z_values = [z, -z]

            for z_val in z_values:
                if len(points) < N:
                    points.append([x, y, z_val])
                else:
                    break

        y = 2 * np.random.rand() - 1
        z = 2 * np.random.rand() - 1

        x4 = 1 - y**4 - z**4

        if x4 >= 0:
            x = x4**(1/4)
            x_values = [x, -x]

            for x_val in x_values:
                if len(points) < N:
                    points.append([x_val, y, z])
                else:
                    break

        z = 2 * np.random.rand() - 1
        x = 2 * np.random.rand() - 1

        y4 = 1 - z**4 - x**4

        if y4 >= 0:
            y = y4**(1/4)
            y_values = [y, -y]

            for y_val in y_values:
                if len(points) < N:
                    points.append([x, y_val, z])
                else:
                    break 
    
    return np.array(points[:N])

def create_pointcloud2(points, frame_id="map"):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]
    
    return pc2.create_cloud(header, fields, points)

def create_mesh_marker(verts, faces, frame_id="map"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 1
    marker.scale.y = 1
    marker.scale.z = 1
    marker.color.a = 0.6
    marker.color.r = 0.99
    marker.color.g = 0.75
    marker.color.b = 0.12

    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1.0

    for face in faces:
        for idx in face:
            pt = verts[idx]
            p = Point()
            p.x, p.y, p.z = pt[0], pt[1], pt[2]
            marker.points.append(p)
    
    return marker

def kompsat_data_publisher(generated_points, verts, faces):
    rospy.init_node('kompsat_data_publisher', anonymous=True)
    pc_pub = rospy.Publisher('/surface_points', PointCloud2, queue_size=10)
    mesh_pub = rospy.Publisher('/surface_mesh', Marker, queue_size=10)
    rate = rospy.Rate(1)
    
    while not rospy.is_shutdown():
        pc_msg = create_pointcloud2(generated_points)
        mesh_msg = create_mesh_marker(verts, faces)
        pc_pub.publish(pc_msg)
        mesh_pub.publish(mesh_msg)
        rospy.loginfo("Published point cloud and mesh marker")
        rate.sleep()

def align_pointcloud_with_pca(points):
    # 중심 이동
    centered = points - np.mean(points, axis=0)

    # PCA (Singular Value Decomposition)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # 회전 행렬
    R = Vt.T
    aligned = np.dot(centered, R)

    return aligned, R, np.mean(points, axis=0)

# 1. 동차 다항식 항 생성 (6차)
def generate_homogeneous_terms(order):
    x, y, z = symbols('x y z')
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            k = order - i - j
            terms.append(x**i * y**j * z**k)
    return terms

# 2. 항 평가 함수
def calculate_polynomial_matrix(points, symb_terms):
    x, y, z = symbols('x y z')
    funcs = [lambdify((x, y, z), term, 'numpy') for term in symb_terms]
    A = np.zeros((points.shape[0], len(symb_terms)))
    for i, func in enumerate(funcs):
        A[:, i] = func(points[:, 0], points[:, 1], points[:, 2])
    return A

# 3. 회귀
def regression_polynomial(points, symb_terms):
    A = calculate_polynomial_matrix(points, symb_terms)
    ones_vec = np.ones((points.shape[0], 1))
    beta = np.linalg.lstsq(A, ones_vec, rcond=None)[0].flatten()
    error = ones_vec.flatten() - np.dot(A, beta)
    return beta, error

# 4. 다항식 수식 구성
def build_polynomial_expression(symb_terms, beta):
    return sum(b * t for b, t in zip(beta, symb_terms))

# 5. 시각화 함수
def visualize_implicit_surface(f_expr, center_shift=None, level=1.0, grid_size=80, grid_range=6):
    x, y, z = symbols('x y z')

    if center_shift is not None:
        f_expr = f_expr.subs({x: x - center_shift[0], y: y - center_shift[1], z: z - center_shift[2]})
        f_expr = expand(f_expr - 1)

    f_func = lambdify((x, y, z), f_expr, modules='numpy')

    # 그리드 생성
    grid_lin = np.linspace(-grid_range, grid_range, grid_size)
    X, Y, Z = np.meshgrid(grid_lin, grid_lin, grid_lin, indexing='ij')
    F = f_func(X, Y, Z)

    # 등가면 추출
    try:
        verts, faces, _, _ = measure.marching_cubes(F, level=0.0)
    except RuntimeError as e:
        print("fail:", e)
        return

    # 좌표 정규화
    scale = (2 * grid_range) / (grid_size - 1)
    verts = verts * scale - grid_range

    # 시각화
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    h = PPm_use.copy()[:,2]
    sc = ax.scatter(PPm_use[:, 0], PPm_use[:, 1], PPm_use[:, 2], c=h, cmap='jet', s=1)
    mesh = Poly3DCollection(verts[faces], alpha=0.6, facecolor=(0.99, 0.75, 0.12))
    ax.add_collection3d(mesh)
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_zlim(-grid_range, grid_range)
    ax.set_title("6 order polynomial regression")
    plt.colorbar(sc)
    plt.tight_layout()
    plt.show()

    return verts, faces

# === 메인 실행 ===

# 예제 입력 (너가 가진 PPm_use를 여기에 넣으면 됨)
# 예: PPm_use = np.loadtxt('pointcloud.txt')  # 외부에서 로드

PP01 = generate_random_points_on_hexagon_prism(3000) + np.random.rand(3000, 3) * 0.0007
PP02 = generate_random_points_on_cube(1000) + np.random.rand(1000, 3) * 0.0005
PP03 = generate_random_points_on_cylinder(70) + np.random.rand(70, 3) * 0.0005
PP04 = generate_random_points_on_cube(200)

shiftReal = [0, 0, -0.01]

PPm_body = PP01.copy()
PPm_body[:,0] = PPm_body[:,0] * 0.576
PPm_body[:,1] = PPm_body[:,1] * 0.576
PPm_body[:,2] = PPm_body[:,2] * 1.165

PPm_panel1 = PP02.copy()
PPm_panel1[:,0] = PPm_panel1[:,0] / 16
PPm_panel1[:,1] = PPm_panel1[:,1] * 1.25
PPm_panel1[:,2] = PPm_panel1[:,2] * 0.576
PPm_panel1 = PPm_panel1 + [0, 2.025, 0]

PPm_panel2 = PP02.copy()
PPm_panel2[:,0] = PPm_panel2[:,0] / 16
PPm_panel2[:,1] = PPm_panel2[:,1] * 1.25
PPm_panel2[:,2] = PPm_panel2[:,2] * 0.576
PPm_panel2 = PPm_panel2 - [0, 2.025, 0]

PPm_box1 = PP04.copy()
PPm_box1[:,0] = PPm_box1[:,0] * 0.333
PPm_box1[:,1] = PPm_box1[:,1] * 0.175
PPm_box1[:,2] = PPm_box1[:,2] * 0.2
PPm_box1 = PPm_box1 + [0, 0.295, -1.365]

PPm_EOC = PP03.copy()
PPm_EOC[:,0] = PPm_EOC[:,0] * 0.09
PPm_EOC[:,1] = PPm_EOC[:,1] * 0.09
PPm_EOC[:,2] = PPm_EOC[:,2] * 0.05
PPm_EOC = PPm_EOC + [-0.115, 0.295, -1.615]

PPm_box2 = PP04.copy()
PPm_box2[:,0] = PPm_box2[:,0] * 0.05
PPm_box2[:,1] = PPm_box2[:,1] * 0.2
PPm_box2[:,2] = PPm_box2[:,2] * 0.1
PPm_box2 = PPm_box2 + [0.2, -0.215, -1.265]

PPm_box3 = PP04.copy()
PPm_box3[:,0] = PPm_box3[:,0] * 0.05
PPm_box3[:,1] = PPm_box3[:,1] * 0.2
PPm_box3[:,2] = PPm_box3[:,2] * 0.2
PPm_box3 = PPm_box3 + [-0.2, -0.215, -1.365]

PPm_total = np.vstack((PPm_body.copy(), PPm_box1.copy(), PPm_EOC.copy(), PPm_box2.copy(), PPm_box3.copy()))
q = [1, 0, 0, 0]
q_scipy = np.roll(q, -1)
rotm = R.from_quat(q_scipy).as_matrix()
PPmR_body = np.dot(PPm_body.copy(), rotm.T)

PPm_use = PPm_body.copy()

# 중심 이동
center_shift = np.median(PPm_use, axis=0)
PPm_shift = PPm_use - center_shift

# 항 생성 및 회귀
order = 6
terms = generate_homogeneous_terms(order)
beta, error = regression_polynomial(PPm_shift, terms)
f_expr = build_polynomial_expression(terms, beta)

min_bounds = np.min(PPm_use, axis=0)
max_bounds = np.max(PPm_use, axis=0)
extent = np.max(max_bounds - min_bounds) / 2

grid_range = extent

# 시각화
verts, faces = visualize_implicit_surface(f_expr, center_shift=center_shift, level=1.0, grid_range=grid_range)

x, y, z = symbols('x y z')
dfdx = diff(f_expr, x)
dfdy = diff(f_expr, y)
dfdz = diff(f_expr, z)
grad_f = lambdify((x, y, z), [dfdx, dfdy, dfdz], 'numpy')

def computer_shift(error, grad):
    A = grad
    b = error[:, np.newaxis]
    shift, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return shift.flatten()

shift_sum = np.zeros(3)
shift_residue_set = []

numIterations = 10
f_numeric = lambdify((x, y, z), f_expr, 'numpy')

for _ in range(numIterations):
    grad_vals = grad_f(PPm_shift[:, 0], PPm_shift[:, 1], PPm_shift[:, 2])
    grad_vals = np.array(grad_vals).T
    error_vector = 1.0 - f_numeric(PPm_shift[:, 0], PPm_shift[:, 1], PPm_shift[:, 2])
    shift = computer_shift(error_vector, grad_vals)
    shift_sum -= shift
    shift_residue_set.append(shift_sum.copy())
    PPm_shift += shift
    A = calculate_polynomial_matrix(PPm_shift, terms)
    beta, _ = regression_polynomial(PPm_shift, terms)
    f_expr = build_polynomial_expression(terms, beta)
    f_numeric = lambdify((x, y, z), f_expr, 'numpy')

f_expr_final = f_expr.subs({x: x - shift_sum[0], y: y - shift_sum[1], z: z - shift_sum[2]})
f_expr_final = expand(f_expr_final - 1)

verts_2, faces_2 = visualize_implicit_surface(f_expr_final)

def save_surface_as_stl(verts, faces, filename="surface_mesh.stl"):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)

save_surface_as_stl(verts_2, faces_2)

try:
    kompsat_data_publisher(PPm_use, verts_2, faces_2)
except rospy.ROSInterruptException:
    pass