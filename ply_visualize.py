import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import open3d as o3d
import numpy as np
import struct

class PlyToPointCloud2Publisher(Node):
    def __init__(self, ply_path, topic_name="ply_cloud", frame_id="map"):
        super().__init__('ply_to_pointcloud2_publisher')
        self.publisher = self.create_publisher(PointCloud2, topic_name, 10)

        # Load .ply file using Open3D
        self.get_logger().info(f"Loading PLY file from: {ply_path}")
        pcd = o3d.io.read_point_cloud(ply_path)
        self.msg = self.convert_o3d_to_pointcloud2(pcd, frame_id)

        # Timer to publish repeatedly
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        self.publisher.publish(self.msg)
        self.get_logger().info("Published PointCloud2 message")

    def convert_o3d_to_pointcloud2(self, pcd, frame_id):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_data = []
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            cloud_data.append(struct.pack('ffffff', x, y, z, r, g, b))

        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 24
        cloud_msg.row_step = 24 * len(points)
        cloud_msg.is_dense = True
        cloud_msg.data = b''.join(cloud_data)

        return cloud_msg

def main():
    rclpy.init()
    ply_path = "merged_icp_result.ply"  # 여기에 실제 파일 경로 입력
    node = PlyToPointCloud2Publisher(ply_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()