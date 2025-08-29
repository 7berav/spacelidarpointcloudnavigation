import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import os
import copy

bag_file = "20250717_133903.bag" # realsense-viewer로 녹화한 .bag 파일
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

pcl_list = []
frame_count = 0

align = rs.align(rs.stream.color)
while True:
    try:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth:
            continue

        pc = rs.pointcloud()
        pc.map_to(depth)
        points = pc.calculate(depth)

        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)

        pcd = pcd.select_by_index([
            i for i, pt in enumerate(pcd.points)
            if np.linalg.norm(pt) < 1.0
        ])

        pcd = pcd.voxel_down_sample(0.01)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(30)

        ply_path = os.path.join(output_dir, f"frame_{frame_count:03d}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        pcl_list.append(pcd)
        frame_count += 1

        if frame_count >= 20:
            break

    except RuntimeError:
        break

pipeline.stop()

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

def clean_and_isolate_object(pcd: o3d.geometry.PointCloud,
                             voxel=0.005,
                             nb_neighbors=20, std_ratio=2.0,
                             plane_dist=0.008, ransac_n=3, num_iter=2000,
                             min_plane_points_ratio=0.02,
                             keep_horizontal=True, keep_vertical=True,
                             horizontal_cos_thresh=0.95,
                             vertical_cos_thresh=0.95,
                             gravity_dir=np.array([0, -1, 0]),
                             view_dir=np.array([0, 0, 1])):
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if len(pcd.points) == 0:
        return pcd

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*6, max_nn=50))
    pcd.normalize_normals()

    total_pts = len(pcd.points)
    working = pcd

    def is_horizontal(normal):
        c = np.abs(float(np.dot(normal, gravity_dir)))
        return c >= horizontal_cos_thresh

    def is_vertical(normal):
        c = np.abs(float(np.dot(normal, view_dir)))
        return c >= vertical_cos_thresh

    for _ in range(8):
        if len(working.points) < 500:
            break
        plane_model, inliers = working.segment_plane(distance_threshold=plane_dist,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iter)
        [a, b, c, d] = plane_model
        n = np.array([a, b, c], dtype=float)
        n = n / (np.linalg.norm(n) + 1e-12)

        inlier_ratio = len(inliers) / max(1, len(working.points))
        if inlier_ratio < min_plane_points_ratio:
            break

        remove_this_plane = False
        if keep_horizontal and is_horizontal(n):
            remove_this_plane = True
        if keep_vertical and is_vertical(n):
            remove_this_plane = True

        if remove_this_plane:
            working = working.select_by_index(inliers, invert=True)
        else:
            break

    if len(working.points) == 0:
        return working
    labels = np.array(working.cluster_dbscan(eps=voxel*6, min_points=50, print_progress=False))
    if labels.max() < 0:
        candidate = working
    else:
        largest_label = int(np.bincount(labels[labels >= 0]).argmax())
        idx = np.where(labels == largest_label)[0].tolist()
        candidate = working.select_by_index(idx)

    pts = np.asarray(candidate.points)
    filtered_idx = np.where(np.linalg.norm(pts, axis=1) > 0.05)[0].tolist()
    candidate = candidate.select_by_index(filtered_idx)

    candidate, _ = candidate.remove_radius_outlier(nb_points=30, radius=voxel*8)
    return candidate

object_only = clean_and_isolate_object(accumulated_pcd,
                                       voxel=0.005,
                                       plane_dist=0.008,
                                       horizontal_cos_thresh=0.93,
                                       vertical_cos_thresh=0.93)

o3d.io.write_point_cloud("merged_icp_object_only.ply", object_only)
print(f"Done. merged_icp_result.ply (전체) / merged_icp_object_only.ply (물체만) 저장됨")