import argparse
from charuco_lidar_calib.lidar_calib import filter_and_calibrate
from charuco_lidar_calib.rosbag import read_rosbag

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rosbag",
        "-b",
        type=str,
        required=True,
        help="Name of rosbag folder in data/rosbags",
    )
    args = parser.parse_args()

    for lidar_points, board_poses in read_rosbag(args.rosbag):
        filter_and_calibrate(lidar_points)
        break
