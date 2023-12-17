"""
parallel.py
Batch CHAT processing utilities 

Arguably, this is entire program equivalent to the original "batchalign"
"""

from batchalign.pipelines import BatchalignPipeline
from batchalign.formats.chat import CHATFile

from threading import Thread
from functools import partial
import os 

def apply_pipeline_file(file, output, pipeline, callback):
       

def apply_pipeline(files, outputs, pipeline, callback, process_count=8):
    threads = []

    for file, output in zip(files, outputs):
        apply_pipeline_file(file, output, pipeline, callback)
