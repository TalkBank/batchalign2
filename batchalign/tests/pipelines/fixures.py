import pytest

from batchalign.pipelines.base import * 

@pytest.fixture(scope="module")
def generator():

    class MyGenerator(BatchalignEngine):
        tasks = [ Task.DEBUG__G ]

        def generate(self, path):
            tmp = Document.new(media_path=path)
            tmp.content.append(Utterance(content="This is a test generation ."))
            tmp.media = Media(type=MediaType.AUDIO, name="generator_wuz_here")

            return tmp

    return MyGenerator()

@pytest.fixture(scope="module")
def processor():

    class MyProcessor(BatchalignEngine):
        tasks = [ Task.DEBUG__P ]

        def process(self, doc):
            doc.content = [Utterance(content="This is a test process .")]

            return doc

    return MyProcessor()

@pytest.fixture(scope="module")
def analyzer():

    class MyAnalyzer(BatchalignEngine):
        tasks = [ Task.DEBUG__A ]

        def analyze(self, doc):
            return doc.model_dump()

    return MyAnalyzer()

