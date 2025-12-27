from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.utils import *
from batchalign.utils.dp import *
from batchalign.constants import *
from batchalign.models.utils import ASRAudioFile

import logging
L = logging.getLogger("batchalign")

import re
import warnings
import tempfile
import os

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import torch
from torchaudio import load
from torchaudio import transforms as T
import torchaudio

class IICFAEngine(BatchalignEngine):
    tasks = [ Task.FORCED_ALIGNMENT ]

    def _hook_status(self, status_hook):
        self.status_hook = status_hook

    def __init__(self):
        self.status_hook = None
        self.__iic = pipeline(
            task=Tasks.speech_timestamp,
            model='iic/speech_timestamp_prediction-v1-16k-offline',
            model_revision="v2.0.4",
            output_dir='./tmp')

    def process(self, doc:Document, **kwargs):
        # check that the document has a media path to align to
        assert doc.media != None and doc.media.url != None, f"We cannot forced-align something that doesn't have a media path! Provided media tier='{doc.media}'"

        if doc.langs[0] not in ["zho", "cmn", "yue"]:
            warnings.warn("Looks like you are not aligning Chinese with IIC; this aligner is designed for Chinese and may not work well with other languages.")

        # load the audio file
        L.debug(f"IIC FA is loading url {doc.media.url}...")
        audio_arr, rate = load(doc.media.url)
        # transpose and mean to get mono
        audio_tensor = torch.mean(audio_arr.transpose(0,1), dim=1)
        audio_file = ASRAudioFile(doc.media.url, audio_tensor, rate)
        L.debug(f"IIC FA finished loading media.")

        # collect utterances into 30 secondish segments to be aligned
        # we have to do this because the aligner does poorly with very short segments
        groups = []
        group = []
        seg_start = 0

        L.debug(f"IIC FA grouping utterances...")

        for i in doc.content:
            if not isinstance(i, Utterance):
                continue
            if i.alignment == None:
                warnings.warn("We found at least one utterance without utterance-level alignment; this is usually not an issue, but if the entire transcript is unaligned, it means that utterance level timing recovery (which is fuzzy using ASR) failed due to the audio clarity. On this transcript, before running forced-alignment, please supply utterance-level links.")
                continue

            # pop the previous group onto the stack
            if (i.alignment[-1] - seg_start) > 15*1000:
                groups.append(group)
                group = []
                seg_start = i.alignment[0]

            # append the contents to the running group
            for word in i.content:
                group.append((word, i.alignment))

        groups.append(group)

        L.debug(f"Begin IIC Inference...")

        for indx, grp in enumerate(groups):
            L.info(f"IIC FA processing segment {indx+1}/{len(groups)}...")
            if self.status_hook != None:
                self.status_hook(indx+1, len(groups))

            # perform alignment
            try:
                # create transcript with spaces between characters
                transcript = []
                for word, _ in grp:
                    # skip punctuation
                    if word.text.strip() not in MOR_PUNCT + ENDING_PUNCT:
                        # add spaces between each character for Chinese
                        transcript.append(" ".join(list(word.text)))

                transcript_text = " ".join(transcript)

                if len(transcript_text.strip()) == 0:
                    continue

                # extract audio chunk and write to temp file
                if (grp[-1][1][1] - grp[0][1][0]) < 20*1000:
                    # get the audio chunk as tensor
                    audio_chunk = audio_file.chunk(grp[0][1][0], grp[-1][1][1])

                    # create temporary file for the audio chunk
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_path = tmp_file.name

                    # write the audio chunk to temp file
                    torchaudio.save(tmp_path, audio_chunk.unsqueeze(0), rate)

                    try:
                        # call IIC aligner with the temp file
                        rec_result = self.__iic(input=(tmp_path, transcript_text),
                                               data_type=("sound", "text"))
                    finally:
                        # clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                else:
                    continue
            except Exception as e:
                L.warning(f"IIC alignment failed for segment {indx+1}: {e}")
                continue

            # parse the result string
            # format: '<sil> 0.000 0.380;一 0.380 0.560;个 0.560 0.800;...'
            try:
                timings = []
                words = []

                for p in rec_result:
                    parts = p["text"].strip().split()
                    timestamps = p["timestamp"]

                    for pts,tss in zip(parts, timestamps):
                        word, start, end = pts, max(tss[1], 0), max(tss[1], 0)
                        words.append(word)
                        # convert to milliseconds and add offset
                        timings.append((int(start + grp[0][1][0]),
                                      int(end + grp[0][1][0])))
            except Exception as e:
                raise e
                L.warning(f"Failed to parse IIC result for segment {indx+1}: {e}")
                continue

            # create reference backplates, which are the word ids to set the timing for
            ref_targets = []
            for indx, (word, _) in enumerate(grp):
                for char in word.text:
                    ref_targets.append(ReferenceTarget(char, payload=indx))

            # create target backplates for the timings
            payload_targets = []
            try:
                for indx, (word, time) in enumerate(zip(words, timings)):
                    for char in word:
                        payload_targets.append(PayloadTarget(char, payload=indx))
            except Exception as e:
                L.warning(f"Failed to create payload targets for segment {indx+1}: {e}")
                continue

            # alignment!
            alignments = align(payload_targets, ref_targets, tqdm=False)

            # set the ids back to the text ids
            # we do this BACKWARDS because we want to have the first timestamp
            # we get about a word first
            alignments.reverse()
            for indx, elem in enumerate(alignments):
                if isinstance(elem, Match):
                    grp[elem.reference_payload][0].time = (timings[elem.payload][0],
                                                           timings[elem.payload][1])

        L.debug(f"Correcting text...")

        # we now set the end alignment of each word to the start of the next
        for doc_ut, ut in enumerate(doc.content):
            if not isinstance(ut, Utterance):
                continue

            # correct each word by bumping it forward
            # and if its not a word we remove the timing
            for indx, w in enumerate(ut.content):
                if w.type in [TokenType.PUNCT, TokenType.FEAT, TokenType.ANNOT]:
                    w.time = None
                elif indx == len(ut.content)-1 and w.text in ENDING_PUNCT:
                    w.time = None
                elif indx != len(ut.content)-1:
                    # search forward for the next compatible time
                    tmp = indx+1
                    while tmp < len(ut.content)-1 and ut.content[tmp].time == None:
                        tmp += 1
                    if w.time == None:
                        continue
                    if ut.content[tmp].time == None:
                        # seek forward one utterance to find their start time
                        next_ut = doc_ut + 1
                        while next_ut < len(doc.content)-1 and (not isinstance(doc.content, Utterance) or doc.content[next_ut].alignment == None):
                            next_ut += 1
                        if next_ut < len(doc.content) and isinstance(doc.content, Utterance) and doc.content[next_ut].alignment:
                            w.time = (w.time[0], doc.content[next_ut].alignment[0])
                        else:
                            w.time = (w.time[0], w.time[0]+500) # give half a second because we don't know

                    # just in case, bound the time by the utterance derived timings
                    if ut.alignment and ut.alignment[0] != None:
                        w.time = (max(w.time[0], ut.alignment[0]), min(w.time[1], ut.alignment[1]))
                    # if we ended up with timings that don't make sense, drop it
                    if w.time and w.time[0] >= w.time[1]:
                        w.time = None

            # clear any built-in timing (i.e. we should use utterance-derived timing)
            ut.time = None
            # correct the text
            if ut.alignment and ut.text != None:
                if '\x15' not in ut.text:
                    ut.text = (ut.text+f" \x15{ut.alignment[0]}_{ut.alignment[1]}\x15").strip()
                else:
                    ut.text = re.sub(r"\x15\d+_\d+\x15",
                                     f"\x15{ut.alignment[0]}_{ut.alignment[1]}\x15", ut.text).strip()
            elif ut.text != None:
                ut.text = re.sub(r"\x15\d+_\d+\x15", f"", ut.text).strip()

        return doc
