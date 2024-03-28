import pyarrow as pa

from dataclasses import dataclass
from enum import Enum


class PredefinedMetadataField(Enum):
    KEY = 0
    ERROR_MESSAGE = 1
    CONTENT_HASH = 2
    IMAGE_WIDTH = 3
    IMAGE_HEIGHT = 4
    ORI_IMAGE_WIDTH = 5
    ORI_IMAGE_HEIGHT = 6

    def pa_data_type(self) -> pa.DataType:
        pa_types = {
            PredefinedMetadataField.IMAGE_WIDTH: pa.int16,
            PredefinedMetadataField.IMAGE_HEIGHT: pa.int16,
            PredefinedMetadataField.ORI_IMAGE_WIDTH: pa.int16,
            PredefinedMetadataField.ORI_IMAGE_HEIGHT: pa.int16,
        }
        return pa_types[self]() if self in pa_types else pa.string()

    def __hash__(self):
        return hash(self.name)


@dataclass
class CustomMetadataField:
    name: str
    pa_dt: pa.DataType

    def pa_data_type(self) -> pa.DataType:
        return self.pa_dt

    def __eq__(self, other):
        if isinstance(other, CustomMetadataField):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)


class Processor:
    SUCCESS_MESSAGE = "Success"

    def __init__(self) -> None:
        self._output_schema = set()

    def pa_metadata_schema(self) -> list[pa.Field]:
        return [pa.field(s.name, s.pa_data_type()) for s in self._output_schema]
