from enum import Enum, IntEnum
from typing import Optional, List, Tuple, Union
from typing_extensions import Annotated

from pydantic import BaseModel, Field, computed_field
from pydantic.functional_validators import BeforeValidator

from batchalign.utils import word_tokenize, sent_tokenize, detokenize

from pathlib import Path

from batchalign.errors import *

class TokenType(IntEnum):
    REGULAR = 0 # hello
    RETRACE = 1 # <I am I am> [/] 
    FEAT = 2 # (.)
    FP = 3 # &-uh
    ANNOT = 4 # &~ject &~head
    PUNCT = 5 # ‡„,
    CORRECTION = 6 # test [= test]

class CustomLineType(IntEnum):
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
    type: TokenType = Field(default=TokenType.REGULAR) # whether the field is a regular word (i.e. not a filled pause, not a feature, not a retrace, etc.)

class Tier(BaseModel):
    lang: str  = Field(default="eng") # eng
    corpus: str = Field(default="corpus_name") # corpus_name
    id: str = Field(default="PAR") # PAR0
    name: str = Field(default="Participant") # Participant

def tokenize_sentence(input):
    if isinstance(input, str):
        words = word_tokenize(input)
        words = [Form(text=i) for i in words]
        return words
    return input
Sentence = Annotated[List[Form], BeforeValidator(tokenize_sentence)]

## TODO: make a computed_field which is the "alignment", which uses
## time if time exists, if not uses the first element of the first utterance
## and the last element of the last utterance

class Utterance(BaseModel):
    tier: Tier = Field(default=Tier())
    content: Sentence
    text: Optional[str] = Field(default=None)
    delim: str = Field(default=".")
    time: Optional[Tuple[int,int]] = Field(default=None)
    custom_dependencies: List[CustomLine]  = Field(default=[])

    @computed_field # type: ignore[misc]
    @property
    def alignment(self) -> Tuple[int,int]:
        if self.time == None: 
            beginning = None
            end = None

            # we scan time forward and backward to get the first set of
            # alignment; this is because we are not exactly sure which of
            # the input words actually carries a time code; for instance
            # utterance beginning don't necessarily have them

            # scan forward to get the time possible
            for i in self.content:
                if i.time:
                    beginning = i.time[0]
                    break

            # scan backward to get the time possible
            for i in reversed(self.content):
                if i.time:
                    end = i.time[-1]
                    break

            
            if beginning == None and end == None:
                return None
            else:
                return (beginning, end)
                
        else: 
            return self.time

    def __getitem__(self, indx):
        return self.content[indx]

    def __len__(self):
        return len(self.content)

    def __str__(self):
        if self.text != None:
            return self.text
        else:
            return self._detokenize()

    def __repr__(self):
        return str(self)

    def _detokenize(self):
        ## TODO deal with angle brackets for retraces
        # NOTE: we don't use detokenize here to put spaces
        # between PUNCT, which should be in CHAT style
        if self.alignment == None:
            return " ".join([i.text for i in self.content])
        else:
            return " ".join([i.text for i in self.content])+f" \x15{str(self.alignment[0])}_{str(self.alignment[1])}\x15"

    def strip(self, join_with_spaces=False):
        """Returns the "core" elements of a sentence, skipping retraces, etc.

        Parameters
        ----------
        join_with_spaces : bool
            Whether to join the simplified utterance with spaces
            instead of treebank detokenization.

        Returns
        -------
        str
            The simplfied utterance.
        """
        
        # filter for words and punctations
        filtered = filter(lambda x:x.type in [TokenType.PUNCT, TokenType.REGULAR],
                          self.content)
        # chain them together
        if join_with_spaces:
            return " ".join([i.text for i in filtered])
        else:
            return detokenize([i.text for i in filtered])
       
class MediaType(str, Enum):
    UNLINKED_AUDIO = "audio, unlinked"
    UNLINKED_VIDEO = "video, unlinked"
    AUDIO = "audio"
    VIDEO = "video"

class Media(BaseModel):
    type: MediaType
    name: str
    url: Optional[str] = Field(default=None)

def tokenize_paragraph(input):
    if isinstance(input, str):
        sentences = sent_tokenize(input)
        sentences = [Utterance(content=i) for i in sentences]
        return sentences
    elif isinstance(input, list) and len(input) > 0 and isinstance(input[0], str):
        sentences = [Utterance(content=i) for i in input]
        return sentences
    return input
Paragraph = Annotated[List[Union[Utterance, CustomLine]], BeforeValidator(tokenize_paragraph)]

class Document(BaseModel):
    content: Paragraph = Field(default=[])
    media: Optional[Media] = Field(default=None)
    langs: List[str] = Field(default=["eng"])

    def __repr__(self):
        return "\n".join(self.transcript())

    def __str__(self):
        return "\n".join(self.transcript())

    def __getitem__(self, indx):
        return self.content[indx]

    def __len__(self):
        return len(self.content)

    @classmethod
    def new(cls, text:Optional[str] = None, media_path:Optional[str] = None, lang:str="eng"):
        # calculate media header, if anything
        media = None
        if media_path:
            media = Media(type=MediaType.UNLINKED_AUDIO,
                          name=Path(media_path).stem,
                          url=media_path)
        # set the content field to be empty, if needed
        if text == None:
            text = []
        # create the doc and set language, if needed
        doc = cls(content=text, media=media)
        if len(doc.tiers) > 0:
            doc.tiers[0].lang = lang

        return doc

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

    @computed_field # type: ignore[misc]
    @property
    def tiers(self) -> List[Tier]:
        results = []
        for i in self.content:
            if isinstance(i, Utterance):
                if i.tier not in results:
                    results.append(i.tier)

        return results

    @tiers.setter
    def tiers(self):
        raise ValueError("Setting `tiers` globally at the document level has unexpected effect and thus is disabled; please set `tier` of each Utterance or change the field of a tier by setting `doc.tiers[n].value = new`.") 


