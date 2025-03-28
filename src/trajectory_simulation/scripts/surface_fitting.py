# 정규화 적용된 surface regression 메인 실행 흐름
import numpy as np
from sympy import symbols, expand, lambdify
from collections import OrderedDict
from sympy import simplify
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_homogeneous_terms(order):
    x, y, z = symbols('x y z')
    terms = []

    for max_power in range(order, -1, -1):
        for j in range(order - max_power, -1, -1):
            k = order - max_power - j

            terms.append(x**max_power * y**j * z**k)
            terms.append(y**max_power * z**j * x**k)
            terms.append(z**max_power * x**j * y**k)

    # 중복 제거 (순서 유지)
    unique_terms = list(OrderedDict.fromkeys([simplify(t) for t in terms]))

    return unique_terms

x, y, z = symbols('x y z')

def calculate_polynomial_matrix(points, symb_terms):
    x, y, z = symbols('x y z')
    funcs = [lambdify((x, y, z), term, 'numpy') for term in symb_terms]
    A = np.zeros((points.shape[0], len(symb_terms)))
    for i, func in enumerate(funcs):
        A[:, i] = func(points[:, 0], points[:, 1], points[:, 2])
    return A

def regression_polynomial(points, symb_terms):
    A = calculate_polynomial_matrix(points, symb_terms)
    ones_vec = np.ones((points.shape[0], 1))
    beta = np.linalg.lstsq(A, ones_vec, rcond=None)[0].flatten()
    error = ones_vec.flatten() - np.dot(A, beta)
    return beta, error

def build_polynomial_expression(symb_terms, beta):
    return sum(b * t for b, t in zip(beta, symb_terms))

def visualize_implicit_surface(f_expr, pointcloud, level=0.0, grid_size=80, grid_range=6):
    x, y, z = symbols('x y z')
    f_func = lambdify((x, y, z), f_expr, modules='numpy')

    # 3D 그리드 생성
    grid_lin = np.linspace(-grid_range, grid_range, grid_size)
    X, Y, Z = np.meshgrid(grid_lin, grid_lin, grid_lin, indexing='ij')
    F = f_func(X, Y, Z)
    F = np.nan_to_num(F, nan=1e6, posinf=1e6, neginf=-1e6)

    # 등가면 추출 (수치 안정성 확인)
    if not (F.min() <= level <= F.max()):
        print(f"⚠️ Level {level} not in range ({F.min()}, {F.max()}) → using median level")
        level = np.median(F)

    verts, faces, _, _ = measure.marching_cubes(F, level=level)

    # 시각화
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
                    c=pointcloud[:, 2], cmap='jet', s=1)
    mesh = Poly3DCollection(verts[faces], alpha=0.6, facecolor=(0.99, 0.75, 0.12))
    ax.add_collection3d(mesh)

    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_zlim(-grid_range, grid_range)
    ax.set_title("6th-order polynomial fitted surface")
    plt.colorbar(sc)
    plt.tight_layout()
    plt.show()

    return verts, faces

# ====================== 데이터 로딩 ======================
data_path = "/home/smrl/cylinder_high.npy"
data = np.load(data_path)
print(f"row {data.shape[0]}, column {data.shape[1]}")
shift_real = np.array([0.5, -1, -0.30])
PPm_use = data + shift_real

# 중심 이동 후 정규화
center_shift = np.mean(PPm_use, axis=0)
scale_shift = np.std(PPm_use, axis=0)
PPm_shift = (PPm_use - center_shift) / scale_shift

numIterations = 10

# 2. 반복 회귀 + gradient shift
shift_sum = np.zeros(3)


# 3. 정규화 좌표계 기준 회귀 완료 후
# 4. 원래 좌표계 기준 shift 복원
shift_total = shift_sum * scale_shift
PPm_aligned = PPm_use - (center_shift + shift_total)

print("mean:", center_shift)
print("std:", scale_shift)

# ====================== 항 생성 및 회귀 ======================
order = 6
terms = generate_homogeneous_terms(order)
A = calculate_polynomial_matrix(PPm_shift, terms)
cond_A = np.linalg.cond(A)
print(f"Condition number of A (normalized): {cond_A}")

beta, error = regression_polynomial(PPm_shift, terms)
f_expr = build_polynomial_expression(terms, beta)

# ====================== 수식 정규화 ======================
# 기대 값 1에 맞게 스케일 조정
f_numeric = lambdify((x, y, z), f_expr, 'numpy')
scale = np.median(f_numeric(PPm_shift[:, 0], PPm_shift[:, 1], PPm_shift[:, 2]))
f_expr_scaled = f_expr / scale
f_expr_final = expand(f_expr_scaled - 1)

# ====================== 좌표 복원 후 시각화 ======================
PPm_aligned = PPm_shift * scale_shift + center_shift

f_test = lambdify((x, y, z), f_expr_final, modules='numpy')
values = f_test(PPm_shift[:, 0], PPm_shift[:, 1], PPm_shift[:, 2])
print("f_expr_final evaluated on normalized pointcloud:")
print("  min =", np.min(values))
print("  max =", np.max(values))
print("  median =", np.median(values))
print("  mean =", np.mean(values))

# 시각화
verts_2, faces_2 = visualize_implicit_surface(f_expr_final, PPm_aligned)

# STL 저장
# save_surface_as_stl(verts_2, faces_2)