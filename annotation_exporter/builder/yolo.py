import warnings
from typing import List

import cv2
import numpy as np

from annotation_exporter.annotations import Task
from annotation_exporter.exporter import Exporter
from .base import Builder
from ..utils import *


def _ls_to_yolo(x1, y1, x2, y2):
    width, height = x2-x1, y2-y1
    x_center, y_center = x1 + width / 2, y1 + height / 2
    return [i / 100 for i in (x_center, y_center, width, height)]


class YoloBuilder(Builder):
    name = "yolo"

    def build_dataset(self, tasks: List[Task], exporters: List[Exporter]):
        for i, task_data in enumerate(tasks):
            if not task_data.annotations:
                continue

            # Download image
            image_bytes = self.s3_context.download_bytes(task_data.image_url)
            image_bytes = np.frombuffer(image_bytes, dtype=np.uint8)

            # Prepare and save image
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            for j, annotation in enumerate(task_data.annotations):
                if not annotation.data_categories:
                    continue

                task_name = f"{i}{j}"
                to_save = rotate_image(image, annotation.image_rotation)
                _, image_bytes = cv2.imencode(".jpg", to_save, [cv2.IMWRITE_JPEG_QUALITY, 100])

                labels = []
                for region in annotation.regions.values():
                    bbox = rotate_ls_box(*region.bounding_box, annotation.image_rotation)
                    bbox = _ls_to_yolo(*bbox)

                    if any(i < 0 or i > 1 for i in bbox):
                        warnings.warn(
                            f"Yolo task {task_name} has values outside [0, 1]",
                            RuntimeWarning
                        )

                    # TODO: We need to get an label map from Label Studio somehow
                    # What's good is that we only use one label and nobody else will ever use this
                    labels.append(f"0 {' '.join(str(i) for i in bbox)}")
                labels_data = "\n".join(labels)

                for e in exporters:
                    if "Training" in annotation.data_categories:
                        e.export_bytes(image_bytes, f"train/images/{task_name}.jpg")
                        e.export_bytes(labels_data.encode("utf-8"), f"train/labels/{task_name}.txt")

                    if "Validation" in annotation.data_categories:
                        e.export_bytes(image_bytes, f"val/images/{task_name}.jpg")
                        e.export_bytes(labels_data.encode("utf-8"), f"val/labels/{task_name}.txt")

            for e in exporters:
                yaml = self._get_yaml()
                e.export_bytes(yaml.encode("utf-8"), "data.yaml")
        
    def _get_yaml(self):
        # TODO: Once again, get label map somehow
        text = "train: ../train/images\nval: ../val/images\n\nnc: 1\nnames: ['Handwriting']"

        return text


__all__ = [
    'YoloBuilder'
]
