from enum import Enum
from typing import Optional, List, Tuple, Union
from typing_extensions import Annotated

from pydantic import BaseModel, Field
from pydantic.functional_validators import BeforeValidator

from .utils import word_tokenize, sent_tokenize, detokenize

class CustomLineType(int, Enum):
    DEPENDENT = 0 # %com
    INDEPENDENT = 1 # @ID

class CustomLine(BaseModel):
    id: str # only the raw string com for %com
    type: CustomLineType # % or @
    content: Optional[str] = Field(default=None) # the contents of the line

class Dependency(BaseModel):
    id: int # first number, 1 indexed
    dep_id: int # second number (where the arrow points to)
    dep_type: str # NSUBJ

class Morphology(BaseModel):
    lemma: str # the lemma
    pos: str # pos like "pron"
    feats: str # string feats "Dem-Acc-S1"

class Form(BaseModel):
    text: str # the text
    time: Optional[Tuple[int, int]] = Field(default=None) # word bullet
    morphology: Optional[List[Morphology]] = Field(default=None) # mor
    dependency: Optional[List[Dependency]] = Field(default=None) # gra

class Tier(BaseModel):
    lang: str # en
    corpus: str # corpus_name
    id: str # PAR0
    name: str # Participant

def tokenize_sentence(input):
    if isinstance(input, str):
        words = word_tokenize(input)
        words = [Form(text=i) for i in words]
        return words
    return input
Sentence = Annotated[List[Form], BeforeValidator(tokenize_sentence)]

class Utterance(BaseModel):
    tier: Tier
    content: Sentence
    text: Optional[str]
    delim: str = Field(default=".")
    alignment: Optional[Tuple[int,int]] = Field(default=None)
    custom_dependencies: List[CustomLine]  = Field(default=[])

    def __str__(self):
        if self.text != None:
            return self.text
        else:
            return self._detokenize()

    def _detokenize(self):
        return detokenize([i.text for i in self.content])

class MediaType(str, Enum):
    UNLINKED_AUDIO = "audio, unlinked"
    UNLINKED_VIDEO = "video, unlinked"
    AUDIO = "audio"
    VIDEO = "video"

class Media(BaseModel):
    type: MediaType
    name: str
    url: Optional[str]

class Document(BaseModel):
    content: List[Union[Utterance, CustomLine]] = Field(default=[])
    media: Optional[Media] = Field(default=None)
    langs: List[str] = Field(default=["eng"])

    def __repr__(self):
        return "Batchalign Document\n-------------------\n"+"\n".join(self.transcript())+"\n-------------------"

    def __str__(self):
        return "\n".join(self.transcript())

    def transcript(self, include_tiers=False, strip=False):
        results = []
        for line in self.content:
            if isinstance(line, Utterance) and strip:
                results.append((line.tier.id+": " if include_tiers
                                else "")+line._detokenize())
            elif isinstance(line, Utterance):
                results.append((line.tier.id+": " if include_tiers
                                else "")+str(line))
            elif line.content != None:
                results.append((line.id+": " if include_tiers
                                else "")+str(line.content))

        return results

    @property
    def tiers(self):
        results = []
        for i in self.content:
            if isinstance(i, Utterance):
                if i.tier not in results:
                    results.append(i.tier)

        return results


