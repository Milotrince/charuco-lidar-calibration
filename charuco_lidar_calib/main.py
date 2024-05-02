from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import numpy as np
import cv2

from charuco_lidar_calib.charuco import get_board_poses

ROSBAG_PATH = "data/rosbags/invert-cube1"
LIDAR_TOPIC = "/luminar_front_points"
IMAGE_TOPIC = "/vimba_front_left_center/image"

bagpath = Path(ROSBAG_PATH)

typestore = get_typestore(Stores.ROS2_IRON)

with AnyReader([bagpath], default_typestore=typestore) as reader:
    for connection, timestamp, rawdata in reader.messages(connections=reader.connections):
         msg = reader.deserialize(rawdata, connection.msgtype)
         if connection.topic == LIDAR_TOPIC:
            print(timestamp, "lidar", msg.data)
         if connection.topic == IMAGE_TOPIC:
            img = msg.data.reshape((1540, 2060, 3))
            poses = get_board_poses(img, show=False)
            print(timestamp, "image", poses)
