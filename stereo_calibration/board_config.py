import cv2

# ChArUco board: 11 columns x 3 rows of squares
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_SIZE = 0.034   # meters (34 mm)
MARKER_SIZE = 0.024   # meters (24 mm)
ARUCO_DICT  = cv2.aruco.DICT_5X5_50


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS), SQUARE_SIZE, MARKER_SIZE, dictionary
    )
    return board, dictionary
