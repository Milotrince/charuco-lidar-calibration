from typing import Tuple
import numpy as np
import open3d as o3d

ICP_THRESHOLD = 10

toCartesian = lambda a: [a[0] / a[2], a[1] / a[2]]
transformation = lambda K, R, t, x: np.dot(K, (t + np.dot(R, x)))

CUBE_LENGTH = 1
REFERENCE_CORNERS = [
    (0.57 / 2 + 0.03, 0.575 / 2 + 0.025, 0),
    (0, 0.57 / 2 + 0.02, 0.56 / 2 + 0.025),
    (0.57 / 2 + 0.03, 0, 0.575 / 2 + 0.02),
]


def reprojection_error(K, R, t, lidar_data, camera_data):
    """
    Compute the reprojection error from 3D Lidar Data to 2D Camera data
    K: 3x4 camera intrinsics matrix
    R: 3x3 rotation matrix
    t: 3x1 translation matrix
    data: 5xn lidar/camera data matrix
    """
    error = 0
    assert len(lidar_data) == len(camera_data)
    for i in range(len(lidar_data)):
        lidar_sample = lidar_data[i]
        camera_sample = camera_data[i]
        computed_data = toCartesian(transformation(K, R, t, lidar_sample))
        diff_x = np.abs(computed_data[0] - camera_sample[0])
        diff_y = np.abs(computed_data[1] - camera_sample[1])
        error += np.linalg.norm(np.array([diff_x, diff_y]))

    return error / len(lidar_data)


def get_icp_transformation_matrix(source, target, init_transformation):
    """
    Uses Iterative Closest Point to match two point clouds together.
    source: source point cloud (open3d.cpu.pybind.geometry.PointCloud)
    target: target point cloud (open3d.cpu.pybind.geometry.PointCloud)
    """
    p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        ICP_THRESHOLD,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000),
    )
    return p2p.transformation


def generate_sample_point_cloud() -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns cube point cloud, along with the midpoints of the AR tags for each plane.

    Returns np.ndarray
    """
    mesh_axis = np.linspace(0, CUBE_LENGTH, num=200)
    mesh_2d = np.stack(np.meshgrid(mesh_axis, mesh_axis), axis=2).reshape(-1, 2)

    total_points = len(mesh_2d)

    xy = np.zeros((total_points, 3))
    xy[:, :2] = mesh_2d
    yz = np.zeros((total_points, 3))
    yz[:, 1:] = mesh_2d
    xz = np.zeros((total_points, 3))
    xz[:, ::2] = mesh_2d

    mesh_cube = np.concatenate([np.array(xy), np.array(yz), np.array(xz)])
    reference_corners = np.array(REFERENCE_CORNERS)

    return mesh_cube, reference_corners

