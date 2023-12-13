from abc import ABC, abstractproperty
from enum import Enum
from typing import List
import copy

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

    def __call__(self, arg):
        arg = copy.deepcopy(arg)
        if len(self.capabilities) == 0:
            raise TypeError(f"Attempted to call default action of an engine that does not report any capabilitie! Engine='{self}', Reported Capabilities='{self.capabilities}'")

        if self.capabilities[0] == BAEngineType.GENERATE:
            return self.generate(arg)
        elif self.capabilities[0] == BAEngineType.PROCESS:
            return self.process(arg)
        elif self.capabilities[0] == BAEngineType.ANALYZE:
            return self.analyze(arg)




