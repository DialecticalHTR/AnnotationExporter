from typing import List 

import numpy as np
import cv2

from annotation_exporter.annotations import Task
from annotation_exporter.exporter import Exporter
from .base import Builder
from ..utils import *


class CraftBuilder(Builder):
    name = "craft"

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
                    x1, y1, x2, y2 = map(lambda c: c / 100, bbox)
                    x1 *= region.original_width
                    x2 *= region.original_width
                    y1 *= region.original_height
                    y2 *= region.original_height
                    bbox = map(int, [x1, y1, x2, y1, x2, y2, x1, y2])

                    labels.append(f"{','.join(str(i) for i in bbox)},{region.text}")
                labels_data = "\n".join(labels)

                for e in exporters:
                    if "Training" in annotation.data_categories:
                        e.export_bytes(image_bytes, f"ch4_training_images/{task_name}.jpg")
                        e.export_bytes(labels_data.encode("utf-8"), f"ch4_training_localization_transcription_gt/gt_{task_name}.txt")

                    if "Validation" in annotation.data_categories:
                        e.export_bytes(image_bytes, f"ch4_test_images/{task_name}.jpg")
                        e.export_bytes(labels_data.encode("utf-8"), f"ch4_test_localization_transcription_gt/gt_{task_name}.txt")

    