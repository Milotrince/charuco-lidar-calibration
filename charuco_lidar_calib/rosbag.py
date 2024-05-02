from pathlib import Path
from typing import Any, Generator, Tuple
import numpy as np

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from charuco_lidar_calib.charuco import get_board_poses
from charuco_lidar_calib.lidar import read_points

ROSBAG_PATH = "data/rosbags/"
LIDAR_TOPIC = "/luminar_front_points"
IMAGE_TOPIC = "/vimba_front_left_center/image"
IMAGE_SHAPE = (1540, 2060, 3)

typestore = get_typestore(Stores.ROS2_IRON)


def read_rosbag(name) -> Generator[Tuple[np.ndarray, Any], None, None]:
    """
    generator that yields (lidar points, charuco board poses)
    """
    bagpath = Path(ROSBAG_PATH + name)
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        last_lidar = None
        last_boardposes = [None, None, None]
        for connection, timestamp, rawdata in reader.messages(
            connections=reader.connections
        ):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == LIDAR_TOPIC:
                points = read_points(msg)
                # print(timestamp, "lidar", list(points))
                last_lidar = np.asarray(list(points))

            if connection.topic == IMAGE_TOPIC:
                img = msg.data.reshape(IMAGE_SHAPE)
                poses = get_board_poses(img, show=False)
                # print(timestamp, "image", poses)
                last_boardposes = [
                    b if b is not None else a for a, b in zip(last_boardposes, poses)
                ]

            if last_lidar is not None and None not in last_boardposes:
                yield (last_lidar, last_boardposes)
                last_lidar = None
                last_boardposes = [None, None, None]
