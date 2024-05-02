from charuco_lidar_calib.camera import *
import cv2

ARUCO_DICT = cv2.aruco.DICT_4X4_250
NUM_ARUCO_PER_BOARD = 4
NUM_ROWS = 3
NUM_COLS = 3
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.02
LENGTH_PX = 300  # total length of the page in pixels
MARGIN_PX = 0  # size of the margin in pixels
FILE_PATH = "data/charuco/charuco_board"
NUM_BOARDS = 3

boards = []
for i in range(NUM_BOARDS):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    dictionary.bytesList = dictionary.bytesList[i * NUM_ARUCO_PER_BOARD :, :, :]
    boards.append(
        cv2.aruco.CharucoBoard(
            (NUM_ROWS, NUM_COLS),
            SQUARE_LENGTH,
            MARKER_LENGTH,
            dictionary,
        )
    )


def get_board_poses(img, show=False):
    poses = []
    for board in boards:
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            img
        )
        if charuco_ids is not None:
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, CAM_MATRIX, CAM_DIST, None, None
            )
            if success:
                poses.append((rvec, tvec))
                if show:
                    cv2.drawFrameAxes(
                        img, CAM_MATRIX, CAM_DIST, rvec, tvec, length=0.1, thickness=15
                    )
                    cv2.imshow("image", img)
                    cv2.waitKey(0)
    return poses


def generate_boards():
    for board in boards:
        img = cv2.aruco.CharucoBoard.generateImage(
            board,
            (LENGTH_PX, int(LENGTH_PX * NUM_COLS / NUM_ROWS)),
            marginSize=MARGIN_PX,
        )
        filename = FILE_PATH + f"-{i+1}.png"
        print("writing to", filename)
        cv2.imwrite(filename, img)


def generate_boards_printable(img):
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.pagesizes import letter
    from tempfile import NamedTemporaryFile

    pieces = []
    for i in range(NUM_COLS):
        for j in range(NUM_ROWS):
            piece = img[
                LENGTH_PX // 3 * i : LENGTH_PX // 3 * (i + 1),
                LENGTH_PX // 3 * j : LENGTH_PX // 3 * (j + 1),
            ]
            pieces.append(piece)

    for i in range(len(pieces)):
        with NamedTemporaryFile(suffix=".png") as f:
            cv2.imwrite(f.name, pieces[i])
            filename = f"{FILE_PATH}/piece-{i}.pdf"
            print("writing to", filename)
            c = canvas.Canvas(filename, pagesize=letter)
            w, h = letter
            margin = 0.5 * inch
            c.drawImage(f.name, margin, h - w + margin, w - 2 * margin, w - 2 * margin)
            c.showPage()
            c.save()


if __name__ == "__main__":
    img = cv2.imread("data/test_img.jpg", flags=cv2.IMREAD_COLOR)
    poses = get_board_poses(img, show=True)
    for i, pose in enumerate(poses):
        print(f"Board {i}: {pose}")
