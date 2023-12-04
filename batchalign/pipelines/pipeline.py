from typing import List, Optional

from batchalign.pipelines.base import *
from batchalign.document import *
from batchalign.errors import *

import logging

L = logging.getLogger('batchalign')

class BatchalignPipeline:
    def __init__(self,
                 generator:Optional[BatchalignEngine] = None,
                 processors:List[BatchalignEngine] = [],
                 analyzer:Optional[BatchalignEngine] = None):
        # check the types of the input
        self.__check_engines(generator, processors, analyzer)

        # store the processors 
        self.__generator = generator
        self.__processors = processors
        self.__analyzer = analyzer

    def __call__(self, input):
        # process input; if its a string, process it as a media
        # only doc seeded from the string. If its not a document,
        # process it as a object to be seeded into a json.
        L.info(f"Pipeline called with engines: generator={self.__generator}, processors={self.__processors}, analyzer={self.__analyzer}")


        L.debug(f"Transforming input of type: {type(input)}")
        doc = input
        if isinstance(input, str):
            if self.__generator:
                doc = Document.new(media_path=input)
            else:
                raise ValueError(f"Pipeline was provided a string input, but is unable to handle generation as no generation engine was provided. Received input: '{input}'")
        elif not isinstance(input, Document):
            doc = Document.model_validate(input)

        L.debug(f"Final input type: {type(input)}")

        # perform input validation
        # checking that the media exists if we have a generator
        if self.__generator and (doc.media == None or doc.media.url == None):
            raise ValueError(f"Generative pipeline was called with no media path!\nHint: did you expect the pipeline to generate a transcript from scratch, and pass in a Document/string that points to a media file PATH with which we can use to generate a transcript?")

        # path processing in sequence
        if self.__generator:
            L.debug(f"Calling generator: {self.__generator}")
            doc = self.__generator.generate(doc.media.url)

        # duplicate the doc
        doc = doc.model_copy(deep=True)

        # perform processing
        for indx, p in enumerate(self.__processors):
            L.debug(f"Calling processor: processor {indx+1}/{len(self.__processors)}, {p}")
            doc = p.process(doc)

        # if needed, perform analysis
        if self.__analyzer:
            L.debug(f"Calling analyzer: {self.__analyzer}")
            return self.__analyzer.analyze(doc)
        else:
            return doc

    @staticmethod
    def __check_engines(generator, processors, analyzer):
        if generator:
            assert BAEngineType.GENERATE in generator.capabilities, f"BatchalignEngine supplied to the pipeline as a generator does not have generation capabilities! Provided engine: {generator}, capabilities: {generator.capabilities}"
        for processor in processors:
            assert BAEngineType.PROCESS in processor.capabilities, f"BatchalignEngine supplied to the pipeline as a processor does not have processing capabilities! Provided engine: {processor}, capabilities: {processor.capabilities}"
        if analyzer:
            assert BAEngineType.ANALYZE in analyzer.capabilities, f"BatchalignEngine supplied to the pipeline as a analyzer does not have analysis capabilities! Provided engine: {analyzer}, capabilities: {analyzer.capabilities}"
            
            




