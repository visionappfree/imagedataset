from collections import namedtuple
from dataclasses import dataclass

Task = namedtuple("Task", ["url", "id", "metadata"], defaults=[{}])
CompletedTask = namedtuple(
    "CompletedTask", ["task", "key", "content", "download_error"]
)


@dataclass
class Data:
    key: str
    content: bytes
    encode_format: str
    metadata: dict
