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
