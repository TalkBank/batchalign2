from enum import Enum, IntEnum
from typing import Optional, List, Tuple, Union, Any, Dict
from typing_extensions import Annotated

from pydantic import BaseModel, Field, computed_field
from pydantic.functional_validators import BeforeValidator

from batchalign.utils import word_tokenize, sent_tokenize, detokenize

from pathlib import Path

from batchalign.errors import *
from batchalign.constants import *

import re

# THE ORDERING OF THESE NUMBERS MATTERS
# this is the order in which the processors are applied
# and even if the user requests a different ordering the
# pipeline WILL apply them in this order. So ensure that
# this order is sensible!
class Task(IntEnum):
    ASR = 3
    UTTERANCE_SEGMENTATION = 4
    SPEAKER_RECOGNITION = 5
    DISFLUENCY_ANALYSIS = 6
    RETRACE_ANALYSIS = 7
    UTTERANCE_TIMING_RECOVERY = 8 # "bulletize"
    FORCED_ALIGNMENT = 9
    FEATURE_EXTRACT = 10
    MORPHOSYNTAX = 11
    COREF = 12
    WER = 13
    TRANSLATE = 14


    DEBUG__G = 0
    DEBUG__P = 1
    DEBUG__A = 2

class TaskType(IntEnum):
    GENERATION = 0
    PROCESSING = 1
    ANALYSIS = 2

TypeMap = {
    Task.ASR: TaskType.GENERATION,
    Task.SPEAKER_RECOGNITION: TaskType.PROCESSING,
    Task.UTTERANCE_SEGMENTATION: TaskType.PROCESSING,
    Task.UTTERANCE_TIMING_RECOVERY: TaskType.PROCESSING,
    Task.FORCED_ALIGNMENT: TaskType.PROCESSING,
    Task.MORPHOSYNTAX: TaskType.PROCESSING,
    Task.FEATURE_EXTRACT: TaskType.ANALYSIS,
    Task.RETRACE_ANALYSIS: TaskType.PROCESSING,
    Task.DISFLUENCY_ANALYSIS: TaskType.PROCESSING,
    Task.COREF: TaskType.PROCESSING,
    Task.WER: TaskType.ANALYSIS,
    Task.TRANSLATE: TaskType.PROCESSING,

    Task.DEBUG__G: TaskType.GENERATION,
    Task.DEBUG__P: TaskType.PROCESSING,
    Task.DEBUG__A: TaskType.ANALYSIS,
}


TaskFriendlyName = {
    Task.ASR: "ASR",
    Task.SPEAKER_RECOGNITION: "Speaker Recognition",
    Task.UTTERANCE_SEGMENTATION: "Utterance Segmentation",
    Task.UTTERANCE_TIMING_RECOVERY: "Utterance Timing Recovery",
    Task.FORCED_ALIGNMENT: "Forced Alignment",
    Task.MORPHOSYNTAX: "Morpho-Syntax",
    Task.FEATURE_EXTRACT: "Feature Extraction",
    Task.RETRACE_ANALYSIS:  "Retrace Analysis",
    Task.DISFLUENCY_ANALYSIS:  "Disfluncy Analysis",
    Task.COREF:  "Coreference Resolution",
    Task.WER:  "Word Error Rate",
    Task.TRANSLATE:  "Translation",
    Task.DEBUG__G:  "TEST_GENERATION",
    Task.DEBUG__P:  "TEST_PROCESSING",
    Task.DEBUG__A:   "TEST_ANALYSIS",
}

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

class Coref(BaseModel):
    start: bool
    end: bool
    chain: int

class Form(BaseModel):
    text: str # the text
    # MILISCEONDS
    time: Optional[Tuple[int, int]] = Field(default=None) # word bullet
    morphology: Optional[List[Morphology]] = Field(default=None) # mor
    dependency: Optional[List[Dependency]] = Field(default=None) # gra
    coreference: Optional[List[Coref]] = Field(default=None) # gra
    type: TokenType = Field(default=TokenType.REGULAR) # whether the field is a regular word (i.e. not a filled pause, not a feature, not a retrace, etc.)

class Tier(BaseModel):
    lang: str  = Field(default="eng") # eng
    corpus: str = Field(default="corpus_name") # corpus_name
    id: str = Field(default="PAR") # PAR0
    name: str = Field(default="Participant") # Participant
    birthday: str = Field(default="") # Participant
    additional: List[str] = Field(default=["","","","",""]) # additional fields 

def get_token_type(str):
    if str in ENDING_PUNCT or str in MOR_PUNCT:
        return TokenType.PUNCT
    else:
        return TokenType.REGULAR

def tokenize_sentence(input):
    if isinstance(input, str):
        words = word_tokenize(input)
        words = [Form(text=i, type=get_token_type(i)) for i in words]
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
    translation: Optional[str] = Field(default=None)
    time: Optional[Tuple[int,int]] = Field(default=None)
    custom_dependencies: List[CustomLine]  = Field(default=[])

    @computed_field
    @property
    def delim(self) -> str:
        if len(self.content) == 0 or self.content[-1].text not in ENDING_PUNCT:
            return '.'
        else:
            return self.content[-1].text

    @property
    def alignment(self) -> Tuple[int,int]:
        # MILISCEONDS
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
            t = self.text
        else:
            t = self._detokenize()

        t = t.replace(". . .", "+...")
        t = t.replace(" ' ", "'")
        t = t.replace("¿", "").replace("¡", "")
        t = re.sub(r"^\+\.\.\.", "", t.strip()).strip()
        # this is here thrice to prevent stuff from not
        # matching once because .sub seems to only match once
        # t = re.sub(r"^[^\w\d\s<]+", "", t.strip()).strip()
        # t = re.sub(r"^[^\w\d\s<]+", "", t.strip()).strip()
        # t = re.sub(r"^[^\w\d\s<]+", "", t.strip()).strip()
        t = re.sub(r",", " , ", t.strip()).strip()
        t = re.sub(r" +", " ", t.strip()).strip()
        t = t.replace("+ ,", "+,").strip()
        return t

    def __repr__(self):
        return str(self)

    def _detokenize(self):
        # create the result by adding minimal CHAT-style annotations
        result = []
        for indx, i in enumerate(self.content):
            if i.type == TokenType.FP:
                result.append("&-"+i.text)
            elif i.type == TokenType.RETRACE:
                # if the next token is a retrace as well, add retrace symbol [/]
                # note that retraces are never the last element in a sentence,
                # this should crash if it
                if indx + 1 >= len(self.content):
                    raise DocumentValidationException(f"Weirdly, a retrace was the last token in an utterance. We cannot parse that.\nHint: there should be another copy of the retrace text after which is the 'regular' text. Check your document format.\nUtterance:{self.strip(True, True, True)}")
                if (self.content[indx+1].type == TokenType.RETRACE and
                    (indx > 0 and self.content[indx-1].type != TokenType.RETRACE) or
                    (indx == 0 and self.content[indx+1].type != TokenType.REGULAR)):
                    if (indx < len(self.content)-1) and self.content[indx+1].text == i.text:
                        result.append(i.text)
                        result.append("[/]")
                        # check if we are about to begin another, seperate retrace
                        # and then mark for that
                        if (indx + 2 < len(self.content) and
                            self.content[indx+2].text != i.text and
                            self.content[indx+2].type == TokenType.RETRACE):
                            result.append("<")
                            
                    else:
                        result.append("<"+i.text)
                elif (self.content[indx+1].type == TokenType.RETRACE and indx > 0 and indx < len(self.content) and
                      self.content[indx+1].text == i.text and self.content[indx-1].text == i.text):
                        result.append(i.text)
                        result.append("[/]")
                        # check if we are about to begin another, seperate retrace
                        # and then mark for that
                        if (indx + 2 < len(self.content) and
                            self.content[indx+2].text != i.text and
                            self.content[indx+2].type == TokenType.RETRACE):
                            result.append("<")
                elif self.content[indx+1].type != TokenType.RETRACE:
                    if indx > 0 and self.content[indx-1].type == TokenType.RETRACE and self.content[indx-1].text != i.text:
                        result.append(i.text+">")
                        result.append("[/]")
                    else:
                        result.append(i.text)
                        result.append("[/]")
                else:
                    result.append(i.text)
            else:
                result.append(i.text)

        # detokenize
        detokenized = " ".join(result)
        # replace suprious spaces caused by edge fudging
        detokenized = re.sub(r"< +", "<", detokenized)
        detokenized = re.sub(r" >", ">", detokenized)

        # check and seperate punct 
        last_tok = result[-1] if len(result) > 0 else ""
        if last_tok in ENDING_PUNCT + MOR_PUNCT:
            detokenized = detokenized.replace(last_tok, f" {last_tok}")
        detokenized = detokenized.replace("  ", " ")
        detokenized = re.sub(r",(\w)", r", \1", detokenized)
        detokenized = re.sub(r",.", r",", detokenized)
        detokenized = re.sub(r".,", r",", detokenized)
        detokenized = re.sub(r"\? !", r"!", detokenized)
        detokenized = re.sub(r"! \?", r"?", detokenized)
        detokenized = detokenized.encode().replace(b"\xef\xbf\xbd", b"").decode("utf-8")
        detokenized = detokenized.encode().replace(b"\xe2\x80\xab", b"").decode("utf-8")

        ## TODO deal with angle brackets for retraces
        # NOTE: we don't use detokenize here to put spaces
        # between PUNCT, which should be in CHAT style
        if self.alignment == None:
            return detokenized
        else:
            return detokenized+f" \x15{str(self.alignment[0])}_{str(self.alignment[1])}\x15"

    def strip(self, join_with_spaces=False, include_retrace=False, include_fp=False):
        """Returns the "core" elements of a sentence, skipping retraces, etc.

        Parameters
        ----------
        join_with_spaces : bool
            Whether to join the simplified utterance with spaces
            instead of treebank detokenization.
        include_retrace : bool
            Whether to include retracing as a part of stripped output.
        include_fp : bool
            Whether to include filled pauses as a part of stripped output.

        Returns
        -------
        str
            The simplfied utterance.
        """
        
        # filter for words and punctations
        to_include = [TokenType.PUNCT, TokenType.REGULAR]
        if include_retrace:
            to_include.append(TokenType.RETRACE)
        if include_fp:
            to_include.append(TokenType.FP)
        filtered = filter(lambda x:x.type in to_include, self.content)
        # chain them together
        if join_with_spaces:
            return " ".join([i.text for i in filtered])
        else:
            return detokenize([i.text for i in filtered])
       
class MediaType(str, Enum):
    UNLINKED_AUDIO = "audio, unlinked"
    UNLINKED_VIDEO = "video, unlinked"
    MISSING_VIDEO = "video, missing"
    MISSING_AUDIO = "audio, missing"
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
    # persistent digital identifier
    pid: Optional[str] = Field(default=None)
    ba_special_: Optional[Dict] = Field(default={})

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

        doc.langs = [lang]

        return doc

    def transcript(self, include_tiers=False, strip=False):
        results = []
        for line in self.content:
            if isinstance(line, Utterance) and strip:
                results.append((line.tier.id+": " if include_tiers
                                else "")+line.strip())
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
                else:
                    # TODO HACKY: this is to ensure that replacing
                    # a single copy of a tier results in all utterancs'
                    # with that same tier being replaced. This is swapping
                    # out the pointers to the underlying tiers to all
                    # point to the same tier instance that's returned
                    i.tier = results[results.index(i.tier)]

        return results

    @tiers.setter
    def tiers(self, x):
        raise ValueError("Setting `tiers` globally at the document level has unexpected effect and thus is disabled; please set `tier` of each Utterance or change the field of a tier by setting `doc.tiers[n].value = new`.") 


