from pathlib import Path

from charuco_lidar_calib.charuco import boards
import cv2
from cv2 import aruco

ROSBAG_PATH = "data/cube1"
LIDAR_TOPIC = "/luminar_front_points"
IMAGE_TOPIC = "/vimba_front_left_center/image"

img = cv2.imread("data/test_img.png", flags=cv2.IMREAD_COLOR)

for board in boards:
   detector = cv2.aruco.CharucoDetector(board)
   charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)
   print("ids", charuco_corners)
   print("corners", charuco_ids)
   print("marker ids", marker_ids)
   # if len(charuco_ids) > 0:
      # Interpolate CharUco corners
      # If enough corners are found, estimate the pose
      # if charuco_retval:
      #    retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)
      #    # If pose estimation is successful, draw the axis
      #    if retval:
      #          cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.1, thickness=15)
