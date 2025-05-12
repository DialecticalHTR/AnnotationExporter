from .base import *

from .trocr import *
from .yolo import *

builders = [
    b.name for b in [TrOCRBuilder, YoloBuilder,]
]
