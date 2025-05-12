from abc import ABC, abstractmethod
from typing import List

from annotation_exporter.s3 import S3Context
from annotation_exporter.exporter import Exporter
from annotation_exporter.annotations import Task


class Builder(ABC):
    def __init__(self, s3_context: S3Context):
        self.s3_context = s3_context

    @abstractmethod
    def build_dataset(tasks: List[Task], exporters: List[Exporter]):
        pass


    @property
    @abstractmethod
    def name(): pass


__all__ = [
    'Builder'
]
