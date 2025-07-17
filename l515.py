# ICP 적용 전체 파이프라인 (정합 안정성 개선 버전)
# .bag 파일로부터 프레임별로 PointCloud 추출하고 ICP 정렬 수행

import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import os
import copy

# === 1. 설정 ===
bag_file = "20250717_133903.bag"  # 예담이의 .bag 파일 경로
output_dir = "frames"  # 추출된 프레임 저장 폴더
os.makedirs(output_dir, exist_ok=True)

# === 2. RealSense 파이프라인 설정 ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# === 3. 프레임 추출 및 PointCloud 생성 ===
pcl_list = []
frame_count = 0

align = rs.align(rs.stream.color)
while True:
    try:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth:
            continue

        # depth to pointcloud
        pc = rs.pointcloud()
        pc.map_to(depth)
        points = pc.calculate(depth)

        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)

        # 필터링 (1m 이내)
        pcd = pcd.select_by_index([
            i for i, pt in enumerate(pcd.points)
            if np.linalg.norm(pt) < 1.0
        ])

        # downsample
        pcd = pcd.voxel_down_sample(0.01)

        # normal 계산
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(30)

        # 저장
        ply_path = os.path.join(output_dir, f"frame_{frame_count:03d}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        pcl_list.append(pcd)
        frame_count += 1

        if frame_count >= 20:
            break

    except RuntimeError:
        break

pipeline.stop()

# === 4. 프레임 간 ICP 정합 ===
transforms = [np.identity(4)]
accumulated_pcd = copy.deepcopy(pcl_list[0])

for i in range(1, len(pcl_list)):
    source = copy.deepcopy(pcl_list[i])
    target = copy.deepcopy(pcl_list[i - 1])

    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    reg = o3d.pipelines.registration.registration_icp(
        source, target, 0.05, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    T = reg.transformation
    transforms.append(transforms[-1] @ T)
    aligned = copy.deepcopy(source).transform(transforms[-1])
    accumulated_pcd += aligned

# === 5. 정합 결과 저장 ===
o3d.io.write_point_cloud("merged_icp_result.ply", accumulated_pcd)
# o3d.visualization.draw_geometries([accumulated_pcd])  # 시각화는 OpenGL 문제 있으면 주석 처리
