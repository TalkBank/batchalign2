import os
import json
import copy
import glob
import tempfile
from pydub import AudioSegment
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.collections.asr.modules.msdd_diarizer import MSDD_module
from batchalign.models.speaker.utils import conv_scale_weights

import torch

# compute device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# override msdd implementation
MSDD_module.conv_scale_weights = conv_scale_weights

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1)

INPUT = "/Users/houjun/Documents/Projects/batchalign2/extern/Untitled.wav"
NUM_SPEAKERS = 2

def resolve_config():
    return os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)), "config.yaml")

base = OmegaConf.load(resolve_config())

# get a input dir
in_file = INPUT
# make a copy of the input config
config = copy.deepcopy(base)
# create a working directory and configure settings
with tempfile.TemporaryDirectory() as workdir:
    # create the mono file
    sound = AudioSegment.from_file(in_file).set_channels(1)
    sound.export(os.path.join(workdir, "mono_file.wav"), format="wav")

    # create configuration with the info we need
    meta = {
        "audio_filepath": os.path.join(workdir, "mono_file.wav"),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
        "num_speakers": NUM_SPEAKERS
    }
    manifest_path = os.path.join(workdir, "input_manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")
    config.diarizer.manifest_filepath = manifest_path
    config.diarizer.out_dir = workdir
    config.device = DEVICE

    # initialize a diarizer and brrr
    msdd_model = NeuralDiarizer(cfg=config)
    msdd_model.diarize()

    # read output and return
    # https://github.com/MahmoudAshraf97/whisper-diarization/blob/main/diarize.py
    speaker_ts = []
    with open(os.path.join(workdir, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])
    print(speaker_ts)



# breakpoint()



# conf = 
# import torch

# torch.rand(1,10).to("mps:0")
