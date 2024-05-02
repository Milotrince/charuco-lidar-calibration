import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from charuco_lidar_calib.icp import generate_sample_point_cloud, get_icp_transformation_matrix


GROUND_ANGLE_THRESH = 10
BACKGROUND_DIST_THRESH = 3.5

CLUSTER_EPS = 0.2
CLUSTER_MIN = 500

OUTLIER_NEIGHBORS = 100
OUTLIER_RADIUS = 0.1

NUM_PLANES = 3
SEGMENT_DIST_THRESH = 0.1
SEGMENT_RANSAC_N = 100
SEGMENT_RANSAC_IT = 1000

DOWNSAMPLE_NUM_SAMPLES = (100**2)*3

SHOW_FILTERED = True


def filter_and_calibrate(pcd_array: np.ndarray):
    pcd = _array_to_pcd(pcd_array)

    # hard-coded filter background
    pcd_without_background, pcd_background = filter_background(pcd)
    pcd_background.paint_uniform_color([1, 0, 0])

    # filter out ground
    pcd_without_ground, pcd_ground = filter_ground(pcd_without_background)
    pcd_ground.paint_uniform_color([1, 1, 0])

    # cluster
    pcd_clustered, pcd_cluster_outliers = filter_cluster(pcd_without_ground)
    pcd_cluster_outliers.paint_uniform_color([1, 0, 1])

    # filter outliers for a final cleanup
    pcd_filtered, pcd_outliers = filter_outliers(pcd_clustered)
    pcd_outliers.paint_uniform_color([0, 0, 1])

    # segment into planes
    # pcd_filtered, pcd_nonplanar = filter_planes(pcd_without_outliers)
    # pcd_nonplanar.paint_uniform_color([1, 0, 0])

    pcd_filtered.paint_uniform_color([0, 1, 0])

    if SHOW_FILTERED:
        o3d.visualization.draw_geometries(
            [pcd_filtered, pcd_outliers, pcd_cluster_outliers, pcd_ground, pcd_background],
        )
    else:
        o3d.visualization.draw_geometries(
            [pcd_filtered],
        )

    pcd_downsampled = pcd_filtered.farthest_point_down_sample(DOWNSAMPLE_NUM_SAMPLES)
    print(len(pcd_downsampled.points))

    # align with ICP
    sample_mesh, sample_reference = generate_sample_point_cloud()
    sample_mesh_pcd = _array_to_pcd(sample_mesh)
    sample_reference_pcd = _array_to_pcd(sample_reference)

    furthest_point = get_furthest_point(pcd_downsampled)

    transform_guess = np.eye(4)
    rotation_guess = o3d.geometry.get_rotation_matrix_from_zyx(np.array([3*np.pi / 4, 0, 0]))
    transform_guess[:3, :3] = rotation_guess
    transform_guess[:3, 3] = furthest_point
    transform = get_icp_transformation_matrix(sample_mesh_pcd, _copy_pcd(pcd_downsampled), transform_guess)

    guess_transformed = _array_to_pcd(sample_mesh).transform(transform_guess)
    guess_transformed.paint_uniform_color([1, 0, 0])
    transformed_mesh = _array_to_pcd(sample_mesh).transform(transform)
    correspondence_pcd = sample_reference_pcd.transform(transform)

    print(transform)
    transformed_mesh.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries(
        [pcd_downsampled, guess_transformed, transformed_mesh]
    )

    return np.asarray(correspondence_pcd.points)



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


def filter_cluster(pcd: o3d.geometry.PointCloud):
    labels = np.array(pcd.cluster_dbscan(eps=CLUSTER_EPS, min_points=CLUSTER_MIN))

    # find largest cluster
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(label_counts)]

    # filter only largest cluster
    filter_indices = np.argwhere(labels == largest_cluster_label)
    filtered_pcd = pcd.select_by_index(filter_indices)
    outliers = pcd.select_by_index(filter_indices, invert=True)

    # max_label = labels.max()
    # print(np.unique(labels))
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return filtered_pcd, outliers
    # return pcd, pcd


def filter_outliers(pcd: o3d.geometry.PointCloud):
    pcd_filtered, inlier_mask = pcd.remove_radius_outlier(
        OUTLIER_NEIGHBORS, OUTLIER_RADIUS
    )
    pcd_outliers = pcd.select_by_index(inlier_mask, invert=True)

    return pcd_filtered, pcd_outliers


# def filter_planes(pcd: o3d.geometry.PointCloud):
#     plane_pcds = []
#
#     cur_pcd = pcd
#     for _ in range(NUM_PLANES):
#         _, inlier_indices = cur_pcd.segment_plane(
#             SEGMENT_DIST_THRESH, SEGMENT_RANSAC_N, SEGMENT_RANSAC_IT
#         )
#         inlier_pcd = cur_pcd.select_by_index(inlier_indices)
#         cur_pcd = cur_pcd.select_by_index(inlier_indices, invert=True)
#
#         plane_pcds.append(inlier_pcd)
#
#     merged_pcds = _array_to_pcd(
#         np.concatenate([np.asarray(plane_pcd.points) for plane_pcd in plane_pcds])
#     )
#
#     return merged_pcds, cur_pcd

def get_furthest_point(pcd: o3d.geometry.PointCloud):
    points = np.asarray(pcd.points)

    distances = np.linalg.norm(points, axis=1)
    return points[np.argmax(distances)]

def _array_to_pcd(arr: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)

    return pcd


def _copy_pcd(pcd: o3d.geometry.PointCloud):
    return o3d.geometry.PointCloud(pcd)
