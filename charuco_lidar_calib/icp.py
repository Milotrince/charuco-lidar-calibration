import numpy as np
import open3d as o3d

ICP_THRESHOLD = 2

toCartesian = lambda a: [a[0] / a[2], a[1] / a[2]]
transformation = lambda K, R, t, x: np.dot(K, (t + np.dot(R, x)))


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
        computed_data = toCartesian(transformation(K, R, t, lidar_data))
        diff_x = np.abs(computed_data[0] - camera_data[0])
        diff_y = np.abs(computed_data[1] - camera_data[1])
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
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )
    return p2p.transformation
