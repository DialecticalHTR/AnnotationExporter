import io
import csv
from typing import List

import cv2
import numpy as np

from annotation_exporter.annotations import Task
from annotation_exporter.exporter import Exporter
from .base import Builder


class YoloBuilder(Builder):
    def build_dataset(self, tasks: List[Task], exporters: List[Exporter]):
        for i, task_data in enumerate(tasks):
            if not task_data.annotations:
                continue

            image_bytes = self.s3_context.download_bytes(task_data.image_url)
            image_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            _, image_bytes = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])

            for e in exporters:
                e.export_bytes(image_bytes, f"train/images/{i}.jpg")

                # TODO: Make an actual train/eval split
                # Janky hack because YOLO requires validation images
                if i == 0:
                    e.export_bytes(image_bytes, f"val/images/{i}.jpg")


            image_height, image_width = image.shape[:2]

            for annotation in task_data.annotations:
                labels = []
                for region in annotation.regions.values():
                    x1, y1, x2, y2 = region.bounding_box
                    width, height = x2-x1, y2-y1
                    x_center, y_center = x1 + width / 2, y1 + height / 2

                    # TODO: We need to get an label map from Label Studio somehow
                    # What's good is that we only use one label and nobody else will ever use this 
                    labels.append(
                        f"0 {x_center / image_width} {y_center / image_height} {width / image_width} {height / image_height}"
                    )

                labels_data = "\n".join(labels)
                e.export_bytes(labels_data.encode("utf-8"), f"train/labels/{i}.txt")
                if i == 0:
                    e.export_bytes(labels_data.encode("utf-8"), f"val/labels/{i}.txt")
            
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
