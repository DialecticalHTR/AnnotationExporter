import abc
import warnings
import dataclasses
from statistics import mean


# @dataclasses.dataclass
# class AnnotationPart:
#     id: str
#     annotation_type: str = dataclasses.field(init=False)
#     from_name: str = dataclasses.field(init=False)
#     to_name: str = dataclasses.field(init=False)


@dataclasses.dataclass
class AnnotationData(abc.ABC):
    id: str

    @staticmethod
    @abc.abstractmethod
    def is_instance(self, types: set[str]) -> bool:
        pass

    @abc.abstractmethod
    def process_part(self, part: dict) -> None:
        pass 


@dataclasses.dataclass
class Region(AnnotationData):
    text: str = dataclasses.field(init=False)
    image_rotation: int | None = dataclasses.field(init=False)
    labels: list[str] = dataclasses.field(default_factory=list)
    points: list[list[int, int]] = dataclasses.field(default_factory=list)

    @staticmethod
    def is_instance(types: set[str]) -> bool:
        return "textarea" in types and ("rectangle" in types or "polygon" in types)
    
    def process_part(self, part):
        annotation_type = part['type']
        value = part['value']

        match annotation_type:
            case 'labels':
                self.labels = value['labels']
                self.image_rotation = part['image_rotation']
            case 'textarea':
                # TODO: make text an list with all text annotations, will do for now
                self.text = value['text'][0]
                self.image_rotation = part['image_rotation']
            case 'rectangle':
                self.type = annotation_type
                x, y, w, h = value['x'], value['y'], value['width'], value['height']

                self.points.append([x, y])
                self.points.append([x+w, y])
                self.points.append([x+w, y+h])
                self.points.append([x, y+h])
                self.points.append(self.points[0])
                
                self.image_rotation = part['image_rotation']
            case 'polygon':
                self.type = annotation_type
                for x, y in value['points']:
                    self.points.append([round(x), round(y)])
                self.points.append(self.points[0])
                
                self.image_rotation = part['image_rotation']
            case _:
                pass
    
    @property
    def bounding_box(self):
        min_x, max_x = min(p[0] for p in self.points), max(p[0] for p in self.points)
        min_y, max_y = min(p[1] for p in self.points), max(p[1] for p in self.points)
        return (min_x, min_y, max_x, max_y)


@dataclasses.dataclass
class Choices(AnnotationData):
    choices: list[str] = dataclasses.field(default_factory=list)

    @staticmethod
    def is_instance(types: set[str]) -> bool:
        return "choices" in types

    def process_part(self, part):
        annotation_type = part['type']

        match annotation_type:
            case 'choices':
                self.choices.extend(part["value"]["choices"])
            case _:
                pass      


@dataclasses.dataclass
class Annotation:
    id: str
    data: dict[str, AnnotationData] = dataclasses.field(default_factory=dict)
    # regions: dict[str, Region] = dataclasses.field(default_factory=dict)
    image_rotation: int = 0

    @classmethod
    def from_json(cls, data) -> "Annotation":
        annotation = cls(id=data['id'])

        # Split parts by id
        data_by_id: dict[str, list[dict]] = {}
        for part in data['result']:
            _id = part['id']
            if _id not in data_by_id:
                data_by_id[_id] = []
            data_by_id[_id].append(part)
        
        # Create annotation data
        for _id, parts in data_by_id.items():
            types = set(part["type"] for part in parts)

            for cls in AnnotationData.__subclasses__():
                if cls.is_instance(types):
                    annotation_data = cls(id=_id)
                    for part in parts:
                        annotation_data.process_part(part)
                    annotation.data[_id] = annotation_data
                    break
            else:
                warnings.warn(f"Failed to find type of annotation data {_id} of annotation {annotation.id}")

        annotation.image_rotation = mean(region.image_rotation for region in annotation.regions.values()) 
        return annotation
    
    @property
    def regions(self):
        return {i: data for i, data in self.data.items() if isinstance(data, Region)}


@dataclasses.dataclass
class Task:
    id: str
    image_url: str = dataclasses.field(init=False)
    annotations: list[Annotation] = dataclasses.field(default_factory=list)


__all__ = [
    'Task',
    'Annotation',
    'Region'
]
