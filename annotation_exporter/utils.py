import cv2
import numpy as np


def rotate_image(image: cv2.Mat, angle: float) -> cv2.Mat:
    rotation_matrix = cv2.getRotationMatrix2D(
        np.array(image.shape[1::-1]) / 2,
        angle,
        1.0
    )
    image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_CUBIC)
    return image


def rotate_point(x, y, angle, origin=(0, 0)) -> tuple[float, float]:
    angle = np.radians(angle)

    x, y = x - origin[0], y - origin[1]
    x = x * np.cos(angle) - y * np.sin(angle)
    y = x * np.sin(angle) + y * np.cos(angle)
    x, y = x + origin[0], y + origin[1]
    return x, y


def rotate_ls_box(x1, y1, x2, y2, angle) -> tuple[float, float, float, float]:
    p1 = rotate_point(x1, y1, angle, (50, 50))
    p2 = rotate_point(x2, y2, angle, (50, 50))

    # get it back to upper-left, lower-right point format
    return min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])


__all__ = [
    "rotate_image",
    "rotate_point",
    "rotate_ls_box"
]
