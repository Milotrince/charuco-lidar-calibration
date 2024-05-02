import numpy as np
import open3d as o3d


GROUND_ANGLE_THRESH = 10
BACKGROUND_DIST_THRESH = 3.5


def filter_and_calibrate(pcd_array: np.ndarray):

    # plane, inliers = pcd.segment_plane(0.1, 100, 10000)
    pcd = _array_to_pcd(pcd_array)

    # hard-coded filter background
    pcd_without_background, pcd_background = filter_background(pcd)
    pcd_background.paint_uniform_color([1.0, 0, 0])

    # filter out ground
    pcd_filtered, pcd_ground = filter_ground(pcd_without_background)
    pcd_ground.paint_uniform_color([1.0, 1.0, 0])

    o3d.visualization.draw_geometries(
        [pcd_filtered, pcd_ground, pcd_background],
    )


def filter_ground(pcd: o3d.geometry.PointCloud):
    o3d_pcd = _copy_pcd(pcd)
    o3d_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
    )
    normals = np.asarray(o3d_pcd.normals)
    angle_threshold = np.cos(np.radians(GROUND_ANGLE_THRESH))  # degree threshold
    dot_up = normals @ np.array([0, 0, 1])
    dot_down = normals @ np.array([0, 0, -1])
    filter_mask = np.logical_and(dot_up < angle_threshold, dot_down < angle_threshold)

    pcd_array = np.asarray(pcd.points)
    filtered_points = pcd_array[~filter_mask]

    remaining_points = pcd_array[filter_mask]
    remaining_normals = normals[filter_mask]
    remaining_pcd = o3d.geometry.PointCloud()
    remaining_pcd.points = o3d.utility.Vector3dVector(remaining_points)
    remaining_pcd.normals = o3d.utility.Vector3dVector(
        remaining_normals
    )  # only for visualization

    return remaining_pcd, _array_to_pcd(filtered_points)


def filter_background(pcd: o3d.geometry.PointCloud):
    pcd_array = np.asarray(pcd.points)

    # filter by normals
    norms = np.linalg.norm(pcd_array, axis=1)
    filter_mask = norms <= BACKGROUND_DIST_THRESH

    background = pcd_array[~filter_mask]
    filtered_points = pcd_array[filter_mask]
    return _array_to_pcd(filtered_points), _array_to_pcd(background)


def _array_to_pcd(arr: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)

    return pcd


def _copy_pcd(pcd: o3d.geometry.PointCloud):
    return o3d.geometry.PointCloud(pcd)
