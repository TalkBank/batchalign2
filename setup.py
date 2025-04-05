import os
from glob import glob
from setuptools import setup, find_packages
from pathlib import Path

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8', errors='ignore').read()

with open(Path("./batchalign") / "version", 'r') as df:
    VERSION_NUMBER, RELEASE_DATE, RELEASE_NOTES = df.readlines()[:3]

setup(
    name = "BatchalignHK",
    author = "Brian MacWhinney, Houjun Liu",
    author_email = "macw@cmu.edu, houjun@cmu.edu",
    version = VERSION_NUMBER.strip(),
    description = ("Python Speech Language Sample Analysis"),
    packages=find_packages(),
    long_description=read('README.md'),
    entry_points = {
        'console_scripts': ['batchalignhk=batchalign.cli.cli:batchalign'],
    },
    long_description_content_type='text/markdown',
    install_requires=[
        "pydantic>=2.4",
        "nltk>=3.8",
        "praatio>=6.0.0,<6.1.0",
        "torch>=2.6.0",
        "torchaudio",
        # "pyAudioAnalysis",
        "hmmlearn==0.3.0",
        "eyed3",
        "opencc-python-reimplemented",
        "pydub",
        "imblearn",
        "plotly>=5.3.0",
        "transformers>=4.37",
        "tokenizers>=0.14.1",
        "pycountry>=22.3",
        "stanza[transformers]>=1.10.1",
        "scipy~=1.11",
        "rev_ai>=2.18.0",
        "rich~=13.6",
        "click~=8.1",
        "matplotlib>=3.8.0,<4.0.0",
        "pyfiglet==1.0.2",
        "setuptools",
        "soundfile~=0.12.0",
        "rich-click>=1.7.0",
        "typing-extensions",
        "num2words",
        "tiktoken",
        "blobfile",
        "sentencepiece",
        "tencentcloud-sdk-python-common",
        "tencentcloud-sdk-python-asr"
    ],
    extras_require={
        'dev': [
            'pytest',
        ],
        'train': [
            'accelerate~=0.27',
        ],
        # 'speaker': [
        #     "nemo-toolkit~=1.21.0",
        #     "omegaconf~=2.3.0",
        #     "pydub~=0.25.0",
        #     "braceexpand",
        #     "editdistance",
        #     "g2p_en",
        #     "ipywidgets",
        #     "jiwer",
        #     "kaldi-python-io",
        #     "kaldiio",
        #     "lhotse>=1.20.0",
        #     "librosa>=0.10.0",
        #     "marshmallow",
        #     "matplotlib",
        #     "packaging",
        #     "pyannote.core",
        #     "pyannote.metrics",
        #     "pydub",
        #     "pyloudnorm",
        #     "resampy",
        #     "ruamel.yaml",
        #     "scipy>=0.14",
        #     "soundfile",
        #     "sox",
        #     "texterrors",
        #     "hydra-core>1.3,<=1.3.2",
        #     "omegaconf<=2.3",
        #     "pytorch-lightning>=2.2.1",
        #     "torchmetrics>=0.11.0",
        #     "transformers>=4.36.0",
        #     "wandb",
        #     "webdataset>=0.2.86",
        #     "sentencepiece",
        #     "youtokentome",
        # ]
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities"
    ]
)


