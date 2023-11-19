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

    def generate(self, source_path: str) -> Document:
        if BAEngineType.GENERATE not in self.capabilities:
            raise TypeError(f"Attempted to use engine that does not have generation capabilities as a generator! Engine='{self}', Reported Capabilities='{self.capabilities}'.")
        pass

    def process(self, doc: Document) -> Document:
        if BAEngineType.PROCESS not in self.capabilities:
            raise TypeError(f"Attempted to use engine that does not have processing capabilities as a processor! Engine='{self}', Reported Capabilities='{self.capabilities}'.")
        pass

    def analyze(self, doc: Document) -> any:
        if BAEngineType.ANALYZE not in self.capabilities:
            raise TypeError(f"Attempted to use engine that does not have analysis capabilities as a analyzer! Engine='{self}', Reported Capabilities='{self.capabilities}'.")
        pass


