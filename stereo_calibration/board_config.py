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


_CHARUCO_DETECTOR = None
_LEGACY_DETECTOR = None
_LEGACY_ANNOUNCED = False


def detect_charuco(gray, board, dictionary, min_corners=6):
    """ChArUco detection compatible with OpenCV >=4.8 (where the free functions
    interpolateCornersCharuco / calibrateCameraCharuco were removed) and older.

    Boards printed with the pre-4.6 API have an INVERTED chessboard phase
    ("legacy pattern"): markers decode fine but no charuco corner interpolates.
    When that signature is seen (many markers, no corners), we retry with a
    legacy-pattern board and stick with whichever works.

    Returns (ch_corners (N,1,2), ch_ids (N,1), marker_corners, marker_ids),
    with ch_corners/ch_ids None when fewer than min_corners are found.
    """
    global _CHARUCO_DETECTOR, _LEGACY_DETECTOR, _LEGACY_ANNOUNCED
    if hasattr(cv2.aruco, "CharucoDetector"):          # OpenCV >= 4.8
        if _CHARUCO_DETECTOR is None:
            _CHARUCO_DETECTOR = cv2.aruco.CharucoDetector(board)
        ch_corners, ch_ids, mk_corners, mk_ids = _CHARUCO_DETECTOR.detectBoard(gray)
        if (ch_ids is None or len(ch_ids) < min_corners) and \
                mk_ids is not None and len(mk_ids) >= 8:
            # markers yes / corners no → try the legacy chessboard phase
            if _LEGACY_DETECTOR is None:
                legacy = cv2.aruco.CharucoBoard(
                    (BOARD_COLS, BOARD_ROWS), SQUARE_SIZE, MARKER_SIZE, dictionary)
                legacy.setLegacyPattern(True)
                _LEGACY_DETECTOR = cv2.aruco.CharucoDetector(legacy)
            c2, i2, m2, mi2 = _LEGACY_DETECTOR.detectBoard(gray)
            if i2 is not None and len(i2) >= min_corners:
                if not _LEGACY_ANNOUNCED:
                    print("[board_config] legacy-pattern ChArUco board detected "
                          "(printed with an old OpenCV) — using setLegacyPattern(True)")
                    _LEGACY_ANNOUNCED = True
                return c2, i2, m2, mi2
        if ch_ids is None or len(ch_ids) < min_corners:
            return None, None, mk_corners, mk_ids
        return ch_corners, ch_ids, mk_corners, mk_ids
    # legacy API (OpenCV < 4.7)
    ad = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    mk_corners, mk_ids, _ = ad.detectMarkers(gray)
    if mk_ids is None or len(mk_ids) < 4:
        return None, None, mk_corners, mk_ids
    retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        mk_corners, mk_ids, gray, board)
    if not retval or retval < min_corners:
        return None, None, mk_corners, mk_ids
    return ch_corners, ch_ids, mk_corners, mk_ids
