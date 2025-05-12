from .base import *

from .trocr import *
from .yolo import *
from .craft import *

builders = [
    TrOCRBuilder, YoloBuilder, CraftBuilder
]
builder_names = [
    b.name for b in builders
]
