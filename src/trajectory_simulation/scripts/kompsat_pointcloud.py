#!/usr/bin/env python3

import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import tf.transformations as tf
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from numpy.polynomial.polynomial import Polynomial
import scipy.optimize as opt
import pandas as pd

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

def pointcloud_publisher(generated_points):
    rospy.init_node('pointcloud_publisher', anonymous=True)
    pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(1)
    
    while not rospy.is_shutdown():
        # points = generate_random_points_on_cylinder(3000)
        points = generated_points
        pc_msg = create_pointcloud2(points)
        pub.publish(pc_msg)
        rospy.loginfo("Published PointCloud2 data")
        rate.sleep()

def nonhomogene_terms(n):
    x, y, z = sp.symbols('x y z')
    terms = []
    for order in range(n, -1, -1):
        for max_power in range(order, -1, -1):
            for j in range(order-max_power, -1, -1):
                k = order - max_power - j
                terms.append(x**max_power * y**j * z**k)
                terms.append(y**max_power * z**j * x**k)
                terms.append(z**max_power * x**j * y**k)
    unique_terms = list(dict.fromkeys(terms))
    return unique_terms

def homogene_terms(order):
    x, y, z = sp.symbols('x y z')
    terms = []
    for max_power in range(order, -1, -1):
        for j in range(order-max_power, -1, -1):
            k = order - max_power - j
            terms.append(x**max_power * y**j * z**k)
            terms.append(y**max_power * z**j * x**k)
            terms.append(z**max_power * x**j * y**k)
    unique_terms = list(dict.fromkeys(terms))
    return unique_terms

# define partial function
def f1_partial(x_val, y_val, z_val):
    return f1_numeric(x_val, y_val, z_val) - 1

def calculate_fourth_order(input_matrix, symbterm):
    # 입력 행렬에서 각 행을 추출하여 각각의 변수 x, y, z로 지정
    x1 = input_matrix[:, 0]
    y1 = input_matrix[:, 1]
    z1 = input_matrix[:, 2]
    
    # 심볼릭 변수 x, y, z 생성
    x, y, z = sp.symbols('x y z')
    num_rows = input_matrix.shape[0]
    num_terms = len(symbterm)
    
    # 출력 행렬 초기화 (n행 m열)
    output = np.zeros((num_rows, num_terms))
    
    # symbterm을 함수로 변환
    term_functions = [sp.lambdify((x, y, z), term, "numpy") for term in symbterm]
    
    # 함수 행렬을 이용하여 모든 행의 값을 한 번에 계산
    for i in range(num_rows):
        for j in range(num_terms):
            output[i, j] = term_functions[j](x1[i], y1[i], z1[i])
    
    return output

def regression_fourth_order(inputMatrix, symbterm):
    Temp = calculate_fourth_order(inputMatrix, symbterm)
    # Temp = np.array(Temp)
    Temp_transpose = Temp.T
    Temp_inv = np.linalg.inv(Temp_transpose @ Temp)
    beta = Temp_inv @ Temp_transpose @ np.ones((inputMatrix.shape[0], 1))
    error = np.ones((inputMatrix.shape[0],1)) - Temp @ beta
    return beta, error

def regressionShift(inputMetrix, errorInput, dxyz):
    beta = np.linalg.inv(dxyz.T @ dxyz) @ (dxyz.T @ errorInput)
    error = errorInput - dxyz @ beta
    return beta, error

if __name__ == '__main__':
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
    PPmR_body = PPm_body.copy() @ rotm.T

    # First plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    h = PPm_total[:,2]
    sc = ax.scatter(PPm_total.copy()[:,0], PPm_total.copy()[:,1], PPm_total.copy()[:,2], c=h, cmap='jet', s=1)
    plt.colorbar(sc)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim([-2,2])
    ax.set_ylim([-4,4])
    ax.set_zlim([-2,2])
    # ax.view_init(azim=45, elev=30)
    plt.title('pointcloud of KOMPSAT-1 model')
    plt.show()

    x, y, z = sp.symbols('x y z')
    order = 6
    PPm_use = PPm_body.copy()

    # 매트랩 상에서 주석 처리 되어있던 코드
    # X_data = PPm_use[:,0]
    # Y_data = PPm_use[:,1]
    # Z_data = PPm_use[:,2]
    # # generate terms
    # termsA_numeric = np.array([[term.subs({x: X, y: Y, z: Z}) for term in termsA] for X, Y, Z in zip(X_data, Y_data, Z_data)])
    # # minimum squares regression
    # beta_values0, residuals, rank, s = np.linalg.lstsq(termsA_numeric, np.ones_link(X_data), rcond=None)
    # # regression model function
    # f1_numeric = sp.lambdify((x, y, z)), sum(b * t for b, t in zip(beta_values0, termsA))

    center_shift = np.median(PPm_use.copy(), axis=0)
    PPm_shift = PPm_use.copy() - center_shift.copy()

    shiftReal = np.array(shiftReal)
    shiftReal_T = shiftReal.copy().reshape(-1, 1)
    shiftResidue = shiftReal_T.copy()
    center_shift_T = center_shift.copy().reshape(-1, 1)
    shiftResidue = shiftResidue.copy() - center_shift_T.copy()

    TermsB = homogene_terms(order)
    betaB = sp.symbols(f'beta0:{len(TermsB)}', real=True)
    f2 = sum(betaB[i] * TermsB[i] for i in range(len(TermsB)))

    difx = sp.diff(f2.copy(), x)
    dify = sp.diff(f2.copy(), y)
    difz = sp.diff(f2.copy(), z)
    f2_numeric = sp.lambdify((x, y, z, *betaB), f2, "numpy")
    d1_numeric = sp.lambdify((x, y, z, *betaB), difx, "numpy")
    d2_numeric = sp.lambdify((x, y, z, *betaB), dify, "numpy")
    d3_numeric = sp.lambdify((x, y, z, *betaB), difz, "numpy")

    beta_values, error = regression_fourth_order(PPm_shift, TermsB)
    error_shift = error.copy()

    # # beta_values = [4.48, 6.26, 0.156, -3.32, -0.17, 0.00480, 0.0206, 0.862, 0.0164, 32.0, -0.555, 0.234, 0.775, -0.675, -0.103, -0.653, 1.138, 0.174, 1.214, -0.0134, -0.1, 2.44, -0.97, -0.103, 1.942, -0.659, -0.00526, -1.895]

    shiftSet = []
    shift1Set = []
    shiftResidueSet = []
    shiftSum = np.vstack((0,0,0))
    numIterations = 10
    shiftSum = center_shift_T.copy()

    # # df = pd.DataFrame(PPm_shift, columns=['x', 'y', 'z'])
    # # df.to_excel("PPm_shift.xlsx", index=False)
    # # print(PPm_shift.shape)

    beta_values_sympy = sp.Matrix(beta_values.copy().T)

    f2_substitued = f2.subs(dict(zip(betaB, beta_values_sympy.copy())))
    f2_substituted_equation = f2_substitued.copy() - 1
    f2_translated = f2_substituted_equation.copy().subs({x: x - center_shift.copy()[0], y: y - center_shift.copy()[1], z: z - center_shift.copy()[2]})
    f2_translated_expanded = sp.expand(f2_translated.copy())

    # # try:
    # #     pointcloud_publisher(PPm_shift)
    # # except rospy.ROSInterruptException:
    # #     pass

    # # Second plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(PPm_use.copy()[:,0], PPm_use.copy()[:,1], PPm_use.copy()[:,2], c=PPm_use.copy()[:,2], cmap='jet', s=1)

    x, y, z = sp.symbols('x y z')

    x_vals = np.linspace(-0.5, 0.5, 50)
    y_vals = np.linspace(-0.5, 0.5, 50)
    X, Y = np.meshgrid(x_vals, y_vals)

    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         print(X[i, j], Y[i, j])

    Z = np.zeros_like(X)

    initial_guesses = np.linspace(-0.5, 0.5, 10)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = sp.solve(f2_translated_expanded.copy().subs({x: X[i, j], y: Y[i, j]}), z)[0]
            except:
                Z[i, j] = np.nan
            # for initial_guess in initial_guesses:
            #     try:
            #         Z[i, j] = sp.nsolve(f2_translated_expanded.copy().subs({x: X[i, j], y: Y[i, j]}), z, initial_guess)
            #         print(initial_guess)
            #         break
            #     except:
            #         Z[i, j] = np.nan
    
    ax.plot_surface(X, Y, Z, color=(0.99, 0.75, 0.12), alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=20, azim=30)
    plt.colorbar(sc)
    plt.title('Pointcloud Model - Panel')
    plt.show()

    # for i in range(numIterations):
    #     # calculate derivative function
    #     dx = np.array([d1_numeric(*point, *beta_values) for point in PPm_shift])
    #     dy = np.array([d2_numeric(*point, *beta_values) for point in PPm_shift])
    #     dz = np.array([d3_numeric(*point, *beta_values) for point in PPm_shift])
    #     dxyz = np.column_stack((dx, dy, dz))

    #     # regressionShift function
    #     shift, residual = regressionShift(PPm_shift, error_shift, dxyz)
    #     shiftResidueSet.append(shift)
    #     shiftSum -= shift
    #     PPm_shift += shift.T
    #     beta_values, error_shift = regression_fourth_order(PPm_shift, TermsB)

    # f3_substituted = f2.subs(dict(zip(betaB, beta_values)))
    # f3_translated = f3_substituted - 1
    # f3_translated = f3_translated.subs({x: x - shiftSum[0], y: y - shiftSum[1], z: z - shiftSum[2]})
    # f3_translated_expanded = sp.expand(f3_translated)

    # # Third plot

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(PPm_use[:,0], PPm_use[:,1], PPm_use[:,2], c=PPm_use[:,2], cmap='jet', s=1)
    # x_vals = np.linspace(-6, 6, 50)
    # y_vals = np.linspace(-6, 6, 50)
    # X, Y = np.meshgrid(x_vals, y_vals)
    # Z = np.zeros_like(X)

    # f3_lambda = sp.lambdify((x, y, z), f3_translated_expanded, "numpy")
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         try:
    #             Z[i, j] = fsolve(lambda z_val: f3_lambda(X[i, j], Y[i, j], z_val), 0)[0]
    #         except:
    #             Z[i, j] = np.nan

    # ax.plot_surface(X, Y, Z, color=(0.05, 0.2, 0.5), alpha=0.5)
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_zlabel('Z (m)')
    # ax.view_init(elev=30, azim=30)
    # plt.colorbar(sc)
    # plt.show()