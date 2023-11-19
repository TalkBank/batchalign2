import pytest

from batchalign.pipelines.pipeline import * 
from batchalign.document import * 

from batchalign.tests.pipelines.fixures import *

PROCESSED_OUTPUT = {'content': [{'tier': {'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR', 'name': 'Participant'}, 'content': [{'text': 'This', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'is', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'a', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'test', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'process', 'time': None, 'morphology': None, 'dependency': None}, {'text': '.', 'time': None, 'morphology': None, 'dependency': None}], 'text': None, 'delim': '.', 'time': None, 'custom_dependencies': [], 'alignment': None}], 'media': {'type': 'audio', 'name': 'generator_wuz_here', 'url': None}, 'langs': ['eng']}
PROCESSED_OUTPUT_GENERATION = {'content': [{'tier': {'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR', 'name': 'Participant'}, 'content': [{'text': 'This', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'is', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'a', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'test', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'generation', 'time': None, 'morphology': None, 'dependency': None}, {'text': '.', 'time': None, 'morphology': None, 'dependency': None}], 'text': None, 'delim': '.', 'time': None, 'custom_dependencies': [], 'alignment': None}], 'media': {'type': 'audio', 'name': 'generator_wuz_here', 'url': None}, 'langs': ['eng']}


def test_standard_pipeline(generator, processor, analyzer):
    pipeline = BatchalignPipeline(generator, [processor], analyzer)
    result = pipeline("path")

    assert PROCESSED_OUTPUT == result

    pipeline = BatchalignPipeline(generator, [], analyzer)
    result = pipeline("path")

    assert PROCESSED_OUTPUT_GENERATION == result

    pipeline = BatchalignPipeline(generator, [processor])
    result = pipeline("path")

    assert Document.model_validate(PROCESSED_OUTPUT) == result

def test_pipeline_with_no_generation(processor):
    pipeline = BatchalignPipeline(processors=[processor])

    with pytest.raises(ValueError):
        pipeline("test")

def test_wrong_pipeline(generator, processor, analyzer):
    with pytest.raises(AssertionError):
        pipeline = BatchalignPipeline(generator=analyzer)

    with pytest.raises(AssertionError):
        pipeline = BatchalignPipeline(processors=[analyzer])

    with pytest.raises(AssertionError):
        pipeline = BatchalignPipeline(analyzer=processor)

