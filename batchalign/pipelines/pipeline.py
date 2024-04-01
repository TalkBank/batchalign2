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

    @staticmethod
    def new(tasks:str, lang="eng", num_speakers=2, **arg_overrides):
        """Create the pipeline.

        Parameters
        ----------
        tasks : str
            The tasks you want the pipeline to do, in a
            comma-seperated list such as `asr,fa,morphosyntax`.
        lang : str
            ISO 3 letter language code.
        num_speakers : int
            Number of speakers.

        kwargs
        ------
        Special package-level overrides.

        Returns
        -------
        BatchalignPipeline
            The pipeline to run.
        """
        
        from batchalign.pipelines.dispatch import dispatch_pipeline
        return dispatch_pipeline(tasks, lang=lang, num_speakers=num_speakers, **arg_overrides)

    def __call__(self, input, callback=None, **kwargs):
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
        L.info(f"Pipeline called with engines: generator={self.__generator}, processors={self.__processors}, analyzer={self.__analyzer} on input of type {type(input)}")

        tt = len(self.__processors) + (0 if self.__generator == None else 1) + (0 if self.__analyzer == None else 1)
        total_tasks = tt

        # call callback, if needed
        if callback:
            callback(0,total_tasks, None)

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
            if callback:
                callback(0,total_tasks, self.__generator.tasks)
            doc = self.__generator.generate(doc.media.url, **kwargs)
            if callback:
                callback(1,total_tasks, self.__generator.tasks)

        # duplicate the doc
        doc = doc.model_copy(deep=True)

        # perform processing
        base = 1 if self.__generator != None else 0
        for indx, p in enumerate(self.__processors):
            L.debug(f"Calling processor: processor {indx+1}/{len(self.__processors)}, {p}")
            if callback:
                callback(base+indx,total_tasks, p.tasks)

            if callback:
                p._hook_status(lambda x,y:callback(base+indx+x,total_tasks+y, p.tasks))
            doc = p.process(doc, **kwargs)
            p._hook_status(None)

            if callback:
                callback(base+indx+1, total_tasks, p.tasks)


        # if needed, perform analysis
        base = (1 if self.__generator != None else 0) + len(self.__processors)
        if self.__analyzer:
            L.debug(f"Calling analyzer: {self.__analyzer}")
            if callback:
                callback(base, total_tasks, self.__analyzer.tasks)

            doc = self.__analyzer.analyze(doc, **kwargs)
            if callback:
                callback(base+1, total_tasks, self.__analyzer.tasks)
            return doc
        else:
            return doc

    @staticmethod
    def __check_engines(engines):
        try:
            capabilities = [i.tasks for i in engines]
        except AttributeError as e:
            raise AttributeError(f"{e}\nPass only initalized engines to BatchalignPipeline!\nHint: did you accidentally call BatchalignPipeline(\"engines,here\") when you meant BatchalignPipeline.new(\"engines,here\")?")

        # we want to ensure that every pipeline has one engine per task
        duplicates = [item for item, count in
                      collections.Counter([i for j in capabilities for i in j]).items()
                      if count > 1]
        if len(duplicates) > 0:
            raise ValueError(f"Pipeline called with engines with overlapping capabilities: duplicate abilities='{duplicates}'!\nIf an engine supports initialization with variadic abilities (i.e. turning off an task it usually performs), please do so.")

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

        
        return generator, sorted(processors, key=lambda x:x.tasks[0]), analyzer, list(sorted(_remove_duplicates([i for j in capabilities for i in j])))

