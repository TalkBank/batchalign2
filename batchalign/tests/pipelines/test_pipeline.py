import pytest

from batchalign.pipelines.pipeline import * 
from batchalign.document import * 

from batchalign.tests.pipelines.fixures import *

PROCESSED_OUTPUT_GENERATION = {'content': [{'tier': {'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR', 'name': 'Participant', 'name': 'Participant', 'birthday': '', 'additional': ['', '', '', '', '']}, 'content': [{'text': 'This', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': 'is', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': 'a', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': 'test', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': 'generation', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': '.', 'time': None, 'morphology': None, 'dependency': None, 'type': 5}], 'text': None, 'delim': '.', 'time': None, 'custom_dependencies': []}], 'media': {'type': 'audio', 'name': 'generator_wuz_here', 'url': None}, 'langs': ['eng'], 'tiers': [{'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR', 'name': 'Participant', 'birthday': '', 'additional': ['', '', '', '', '']}], "pid": None}

PROCESSED_OUTPUT = {'content': [{'tier': {'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR', 'name': 'Participant', 'name': 'Participant', 'birthday': '', 'additional': ['', '', '', '', '']}, 'content': [{'text': 'This', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': 'is', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': 'a', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': 'test', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': 'process', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': '.', 'time': None, 'morphology': None, 'dependency': None, 'type': 5}], 'text': None, 'delim': '.', 'time': None, 'custom_dependencies': []}], 'media': {'type': 'audio', 'name': 'generator_wuz_here', 'url': None}, 'langs': ['eng'], 'tiers': [{'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR', 'name': 'Participant', 'birthday': '', 'additional': ['', '', '', '', '']}], "pid": None}

MODEL_NO_MEDIA = {'content': [{'tier': {'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR', 'name': 'Participant', 'name': 'Participant', 'birthday': '', 'additional': ['', '', '', '', '']}, 'content': [{'text': 'This', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'is', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'a', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'test', 'time': None, 'morphology': None, 'dependency': None}, {'text': 'process', 'time': None, 'morphology': None, 'dependency': None}, {'text': '.', 'time': None, 'morphology': None, 'dependency': None, 'type': 5}], 'text': None, 'delim': '.', 'time': None, 'custom_dependencies': []}], 'langs': ['eng'], 'tiers': [{'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR', 'name': 'Participant', 'birthday': '', 'additional': ['', '', '', '', '']}]}


def test_standard_pipeline(generator, processor, analyzer):
    pipeline = BatchalignPipeline(generator, processor, analyzer)
    result = pipeline("path")

    assert result == PROCESSED_OUTPUT

    pipeline = BatchalignPipeline(generator, analyzer)
    result = pipeline("path")

    assert PROCESSED_OUTPUT_GENERATION == result

    pipeline = BatchalignPipeline(generator, processor)
    result = pipeline("path")

    assert Document.model_validate(PROCESSED_OUTPUT) == result

def test_pipeline_with_no_generation(processor):
    pipeline = BatchalignPipeline(processor)

    with pytest.raises(ValueError):
        pipeline("test")

def test_pipeline_that_cant_generate(generator):
    pipeline = BatchalignPipeline(generator)

    nomedia = Document.model_validate(MODEL_NO_MEDIA)
    nomediapath = Document.model_validate(PROCESSED_OUTPUT)

    with pytest.raises(ValueError):
        pipeline(nomedia)
    with pytest.raises(ValueError):
        pipeline(nomediapath)

def test_wrong_pipeline(generator, processor, analyzer):
    with pytest.raises(ValueError):
        pipeline = BatchalignPipeline(generator, generator)

    with pytest.raises(ValueError):
        pipeline = BatchalignPipeline(analyzer, generator, processor, processor)

def test_pipeline_error_creation():
    with pytest.raises(ValueError):
        pipeline = BatchalignPipeline.new("chicken")
