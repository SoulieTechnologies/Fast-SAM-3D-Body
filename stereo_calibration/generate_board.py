"""
Generate the ChArUco 11x3 calibration board image.
Print this at the correct physical size (measure SQUARE_SIZE after printing).
"""

import cv2
import board_config

board, _ = board_config.make_board()

# 100 px per square → 1100 x 300 px + 20 px margin on each side
px_per_square = 100
W = board_config.BOARD_COLS * px_per_square + 40
H = board_config.BOARD_ROWS * px_per_square + 40

img = board.generateImage((W, H), marginSize=20, borderBits=1)
out = "calibration_data/charuco_11x3.png"
cv2.imwrite(out, img)
print(f"Board saved to {out}  ({W}x{H} px)")
print(
    f"Each square = {board_config.SQUARE_SIZE*1000:.0f} mm when printed at "
    f"{px_per_square} px/sq — adjust your printer scaling accordingly."
)
