import numpy as np
import cv2

ARUCO_DICT = cv2.aruco.DICT_4X4_250
NUM_ARUCO_PER_BOARD = 4
NUM_ROWS = 3
NUM_COLS = 3
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.02
LENGTH_PX = 300  # total length of the page in pixels
MARGIN_PX = 0  # size of the margin in pixels
FILE_PATH = "../data/charuco/charuco_board"
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

if __name__ == "__main__":
    for board in boards:
        img = cv2.aruco.CharucoBoard.generateImage(
            board,
            (LENGTH_PX, int(LENGTH_PX * NUM_COLS / NUM_ROWS)),
            marginSize=MARGIN_PX,
        )
        cv2.imwrite(FILE_PATH + f"-{i+1}.png", img)

    # FOR PRINTING

    # from reportlab.pdfgen import canvas
    # from reportlab.lib.units import inch
    # from reportlab.lib.pagesizes import letter
    # from tempfile import NamedTemporaryFile

    # for i in range(len(pieces)):
    #     with NamedTemporaryFile(suffix=".png") as f:
    #         cv2.imwrite(f.name, pieces[i])

    #         c = canvas.Canvas(f"piece-{i}.pdf", pagesize=letter)
    #         w, h = letter
    #         margin = 0.5 * inch
    #         c.drawImage(f.name, margin, h-w + margin, w - 2*margin, w - 2*margin)
    #         c.showPage()
    #         c.save()
