import os
import json
import pathlib

import pytest

from batchalign.document import Document

@pytest.fixture(scope="package")
def en_doc():
    """Tagged gold English document.

    This is a gold tagged document, which passes CHATTER and CHECK.
    The purpose of this document is to be taken and re-tagged by
    various pipelines to check whether it is behaving in the way
    that we expect. For instance, you can pass it through UD again
    to see if the tagging performs in the way that we want by checking
    if the document remains the same before and after UD.

    Returns
    -------
    Document
        The gold document.
    """
    
    dir = pathlib.Path(__file__).parent.resolve()

    with open(os.path.join(dir, "support", "test.json"), 'r') as df:
        doc = Document.model_validate(json.load(df))

    doc.media.url = os.path.join(dir, "support", "test.mp3")

    return doc

@pytest.fixture(scope="package")
def en_utterance(en_doc):
    """Tagged gold English utterance.

    This is a gold tagged document, which passes CHATTER and CHECK.
    The purpose of this document is to be taken and re-tagged by
    various pipelines to check whether it is behaving in the way
    that we expect. For instance, you can pass it through UD again
    to see if the tagging performs in the way that we want by checking
    if the document remains the same before and after UD.

    Returns
    -------
    Utterance
        The gold utterance.
    """

    return en_doc[0]
   
