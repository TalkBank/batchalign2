from typing import List, Optional

from batchalign.pipelines.base import *
from batchalign.document import *
from batchalign.errors import *
import collections

import logging

L = logging.getLogger('batchalign')

def _remove_duplicates(x):
    a = []
    for i in x:
        if i not in a:
            a.append(i)
    return a

class BatchalignPipeline:
    def __init__(self, *engines):
        # check the types of the input
        generator, processors, analyzer, capabilities = self.__check_engines(engines)

        # store the processors 
        self.__generator = generator
        self.__processors = processors
        self.__analyzer = analyzer
        self.__capabilities = capabilities

    @property
    def tasks(self):
        return self.__capabilities

    def __call__(self, input, callback=None):
        """Call the pipeline.

        Parameters
        ----------
        input : any
            The input to the pipeline.
        callback : callable
            The callback function to send a new tick to for progress reporting, if any.

        Return
        ------
        Document
            The Processed document.
        """
        
        # process input; if its a string, process it as a media
        # only doc seeded from the string. If its not a document,
        # process it as a object to be seeded into a json.
        L.info(f"Pipeline called with engines: generator={self.__generator}, processors={self.__processors}, analyzer={self.__analyzer}")

        counter = 0 
        total_tasks = len(self.__processors) + 0 if self.__generator == None else 1 + 0 if self.__analyzer == None else 1

        # call callback, if needed
        if callback:
            callback(counter,total_tasks, None)

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
            counter += 1
            if callback:
                callback(counter,total_tasks, self.__generator.tasks)

        # duplicate the doc
        doc = doc.model_copy(deep=True)

        # perform processing
        for indx, p in enumerate(self.__processors):
            L.debug(f"Calling processor: processor {indx+1}/{len(self.__processors)}, {p}")
            doc = p.process(doc)

            counter += 1
            if callback:
                callback(counter, total_tasks, p.tasks)


        # if needed, perform analysis
        if self.__analyzer:
            L.debug(f"Calling analyzer: {self.__analyzer}")
            doc = self.__analyzer.analyze(doc)
            counter += 1
            if callback:
                callback(counter, total_tasks, self.__analyzer.tasks)
            return doc
        else:
            return doc

    @staticmethod
    def __check_engines(engines):
        capabilities = [i.tasks for i in engines]

        # we want to ensure that every pipeline has one engine per task
        duplicates = [item for item, count in
                      collections.Counter([i for j in capabilities for i in j]).items()
                      if count > 1]
        if len(duplicates) > 0:
            raise ValueError(f"Pipeline called with engines with overlapping capabilities: duplicate abilities='{duplicates}'!\nIf an engine supports initialization with variadic abilities (i.e. turning off an task it usual performs), please do so.")

        # we want to make sure we only have one generator and one analyzer
        # and we want to construct the rest based on the order they were provided
        generator = None
        processors = []
        analyzer = None

        for i in engines:
            if TaskType.GENERATION in [TypeMap.get(j) for j in i.tasks] and generator == None:
                generator = i
            elif TaskType.GENERATION in [TypeMap.get(j) for j in i.tasks]: 
                raise ValueError(f"Multiple generators found for the same pipeline! We don't know which one to start with. Current generator = '{generator}' conflicts with other generator = '{i}'")
            elif TaskType.ANALYSIS in [TypeMap.get(j) for j in i.tasks] and analyzer == None: 
                analyzer = i
            elif TaskType.ANALYSIS in [TypeMap.get(j) for j in i.tasks]: 
                raise ValueError(f"Multiple analyzers found for the same pipeline! We don't know which one to end with. Current analyzer = '{analyzer}' conflicts with other analyzer = '{i}'")
            elif TaskType.PROCESSING in [TypeMap.get(j) for j in i.tasks]: 
                processors.append(i)
            else:
                raise ValueError(f"Engine provided to pipeline with no apparent purpose (i.e. its not a generator, processor, nor analyzer). Engine = '{i}'")

        return generator, processors, analyzer, list(sorted(_remove_duplicates([i for j in capabilities for i in j])))

