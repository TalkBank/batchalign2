import stanza
from batchalign.utils.dp import PayloadTarget, ReferenceTarget, Match, align
from warnings import warn
from batchalign.document import *
from batchalign.constants import *
from batchalign.pipelines.base import *
from batchalign.formats.chat.parser import chat_parse_utterance

from batchalign.utils.dp import *



class CorefEngine(BatchalignEngine):
    tasks = [ Task.COREF ]

    def process(self, doc, **kwargs):
        if "eng" not in doc.langs:
            warn("Coreference resolution is only supported for English documents.")
            return 

        detokenized = " ".join([i.strip(include_retrace=True, include_fp=True) for i in doc.content if isinstance(i, Utterance)])
        pipeline = stanza.Pipeline(lang="en", processors="tokenize, coref", package={"coref": "ontonotes-singletons_roberta-large-lora"})

        coref_chains = pipeline(detokenized).sentences
        coref_chains = [(j.text,
                        [Coref(start=chain.is_start,
                                end=chain.is_end,
                                chain=chain.chain.index) for chain in j.coref_chains])
                        for i in coref_chains
                        for j in i.words]

        payloads = [PayloadTarget(i[0], i[1]) for i in coref_chains]
        references = [ReferenceTarget(j.text, (ut_id, form_id)) for ut_id, i in enumerate(doc.content)
                        if isinstance(i, Utterance)
                        for form_id, j in enumerate(i.content)]
        alignment = align(payloads, references, tqdm=False)

        for i in alignment:
            if isinstance(i, Match):
                (ut, form) = i.reference_payload
                doc.content[ut].content[form].coreference = i.payload

        return doc


