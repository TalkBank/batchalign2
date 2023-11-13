from abc import ABC, abstractproperty
from enum import Enum
from typing import List

from batchalign.document import *

class BAEngineType(Enum):
    GENERATE = 0
    PROCESS = 1
    ANALYZE = 2

class BatchalignEngine(ABC):

    @abstractproperty
    def capabilities(self) -> List[BAEngineType]:
        pass

    def generate(self, source_path: str, *args, **kwargs) -> Document:
        pass

    def process(self, doc: Document, *args, **kwargs) -> Document:
        pass

    def analyze(self, doc: Document, *args, **kwargs) -> any:
        pass


