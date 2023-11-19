import pytest

from batchalign.document import * 
from batchalign.tests.pipelines.fixures import *

def test_use_engine_wrong(processor, generator, analyzer):
    # check that the engines complain when they are used wrong

    with pytest.raises(TypeError):
        processor.generate(None)
        processor.analyze(None)

    with pytest.raises(TypeError):
        generator.analyze(None)
        generator.process(None)

    with pytest.raises(TypeError):
        analyzer.generate(None)
        analyzer.process(None)



