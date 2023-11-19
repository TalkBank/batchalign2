import pytest

from batchalign.pipelines.base import * 

@pytest.fixture(scope="module")
def generator():

    class MyGenerator(BatchalignEngine):
        capabilities = [ BAEngineType.GENERATE ]

        def generate(self, path):
            tmp = Document.from_media(path)
            tmp.content.append(Utterance(content="This is a test generation ."))
            tmp.media = Media(type=MediaType.AUDIO, name="generator_wuz_here")

            return tmp

    return MyGenerator()

@pytest.fixture(scope="module")
def processor():

    class MyProcessor(BatchalignEngine):
        capabilities = [ BAEngineType.PROCESS ]

        def process(self, doc):
            doc.content = [Utterance(content="This is a test process .")]

            return doc

    return MyProcessor()

@pytest.fixture(scope="module")
def analyzer():

    class MyAnalyzer(BatchalignEngine):
        capabilities = [ BAEngineType.ANALYZE ]

        def analyze(self, doc):
            return doc.model_dump()

    return MyAnalyzer()

