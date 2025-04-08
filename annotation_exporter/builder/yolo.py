from typing import List

import cv2
import numpy as np

from annotation_exporter.annotations import Task
from annotation_exporter.exporter import Exporter
from .base import Builder
from ..utils import rotate_image


def _rotate_point(x, y, angle, origin=(0, 0)) -> tuple[float, float]:
    angle = np.radians(angle)

    x, y = x - origin[0], y - origin[1]
    x = x * np.cos(angle) - y * np.sin(angle)
    y = x * np.sin(angle) + y * np.cos(angle)
    x, y = x + origin[0], y + origin[1]
    return x, y


def _rotate_yolo_box(x1, y1, x2, y2, angle) -> tuple[float, float, float, float]:
    return *_rotate_point(x1, y1, angle, (0.5, 0.5)), *_rotate_point(x2, y2, angle, (0.5, 0.5))


def _ls_to_yolo(x1, y1, x2, y2):
    return [i / 100 for i in (x1, y1, x2, y2)]


class YoloBuilder(Builder):
    def build_dataset(self, tasks: List[Task], exporters: List[Exporter]):
        validation_image_saved = False
        validation_annotation_saved = False

        for i, task_data in enumerate(tasks):
            if not task_data.annotations:
                continue

            # Download image
            image_bytes = self.s3_context.download_bytes(task_data.image_url)
            image_bytes = np.frombuffer(image_bytes, dtype=np.uint8)

            # Prepare and save image
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            for j, annotation in enumerate(task_data.annotations):
                task_name = f"{i}{j}"
                to_save = rotate_image(image, annotation.image_rotation)
                _, image_bytes = cv2.imencode(".jpg", to_save, [cv2.IMWRITE_JPEG_QUALITY, 100])

                for e in exporters:
                    e.export_bytes(image_bytes, f"train/images/{task_name}.jpg")

                # TODO: Make an actual train/eval split
                # Janky hack because YOLO requires validation images
                if not validation_image_saved:
                    e.export_bytes(image_bytes, f"val/images/{task_name}.jpg")
                    validation_image_saved = True

                labels = []
                for region in annotation.regions.values():
                    bbox = _ls_to_yolo(*region.bounding_box)
                    x1, y1, x2, y2 = _rotate_yolo_box(*bbox, annotation.image_rotation)

                    width, height = x2 - x1, y2 - y1
                    x_center, y_center = x1 + width / 2, y1 + height / 2

                    # TODO: We need to get an label map from Label Studio somehow
                    # What's good is that we only use one label and nobody else will ever use this 
                    labels.append(
                        f"0 {x_center} {y_center} {width} {height}"
                    )

                labels_data = "\n".join(labels)
                e.export_bytes(labels_data.encode("utf-8"), f"train/labels/{task_name}.txt")
                if not validation_annotation_saved:
                    e.export_bytes(labels_data.encode("utf-8"), f"val/labels/{task_name}.txt")
                    validation_annotation_saved = True
            
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
