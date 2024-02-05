from abc import ABC, abstractproperty
from enum import Enum
from typing import List
import copy

from batchalign.document import *

class BatchalignEngine(ABC):

    @abstractproperty
    def tasks(self) -> List[Task]:
        pass

    def generate(self, source_path: str, **kwargs) -> Document:
        raise NotImplementedError(f"Attempted to use an engine as a generator that didn't implement a generation method! Engine='{self}', Reported Tasks='{self.tasks}'.")
        pass

    def process(self, doc: Document, **kwargs) -> Document:
        raise NotImplementedError(f"Attempted to use an engine as a processor that didn't implement a processing method! Engine='{self}', Reported Tasks='{self.tasks}'.")
        pass

    def analyze(self, doc: Document, **kwargs) -> dict:
        raise NotImplementedError(f"Attempted to use an engine as an analyzer that didn't implement a analysis method! Engine='{self}', Reported Tasks='{self.tasks}'.")
        pass

    def __call__(self, arg, **kwargs):
        arg = copy.deepcopy(arg)
        if len(self.tasks) == 0:
            raise TypeError(f"Attempted to call default action of an engine that does not report any capabilities! Engine='{self}', Reported Capabilities='{self.tasks}'")

        if TypeMap.get(self.tasks[0]) == TaskType.GENERATION:
            return self.generate(arg, **kwargs)
        elif TypeMap.get(self.tasks[0]) == TaskType.PROCESSING:
            return self.process(arg, **kwargs)
        elif TypeMap.get(self.tasks[0]) == TaskType.ANALYSIS:
            return self.analyze(arg, **kwargs)
        else:
            raise TypeError(f"Attempted to use the default task of an engine whose default task is uninterpretable! Engine='{self}', Default Task='{self.tasks[0]}'")

    def _hook_status(self, status_hook):
        # get current status of the engine; i.e. "processed n out of n"
        # any engines that support status hooking will call status_hook
        # to update status
        return
        




