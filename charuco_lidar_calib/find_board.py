from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import numpy
import cv2
from cv2 import aruco

ROSBAG_PATH = "data/cube1"
LIDAR_TOPIC = "/luminar_front_points"
IMAGE_TOPIC = "/vimba_front_left_center/image"
# CHARCUO_ROWS = 3
# CHARCUO_COLS = 3
CHARUCO_DICT = aruco.DICT_4X4_250

dictionary = aruco.getPredefinedDictionary(CHARUCO_DICT)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

bagpath = Path(ROSBAG_PATH)

typestore = get_typestore(Stores.ROS2_IRON)

with AnyReader([bagpath], default_typestore=typestore) as reader:
    for connection, timestamp, rawdata in reader.messages(connections=reader.connections):
         msg = reader.deserialize(rawdata, connection.msgtype)
         if connection.topic == LIDAR_TOPIC:
            print(timestamp, "lidar", msg.data)
         if connection.topic == IMAGE_TOPIC:
            img = msg.data
            # markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)
            out = detector.detectMarkers(img)
            print(timestamp, "image", out)
