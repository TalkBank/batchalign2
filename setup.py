import os
from glob import glob
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "batchalign",
    author = "Brian MacWhinney, Houjun Liu",
    author_email = "macw@cmu.edu, houjun@cmu.edu",
    version = "2.0.0",
    description = ("Python Speech Language Sample Analysis"),
    packages=find_packages(),
    package_data={
        'batchalign': [os.path.basename(i)
                       for i in glob(os.path.join("batchalign", "pipelines", "cleanup",
                                                  "support", "*"))],
    },
    install_requires=[
        "pydantic>=2.4",
        "nltk>=3.8",
        "montreal-forced-aligner>=3.0.0",
        "praatio>=6.0.0,<6.1.0",
        "pytorch>=2.1.0,<2.2.0",
        "torchaudio>=2.1.0,<2.2.0",
        "pyAudioAnalysis==0.3.14",
        "hmmlearn==0.3.0",
        "eyed7>=0.9.7",
        "pydub>=0.25.1,<0.26.0",
        "imblearn",
        "plotly>=5.18.0",
        "transformers>=4.35",
        "tokenizers>=0.14.1",
        "ffmpeg-python==0.2.0",
        "pycountry>=22.3",
        "stanza>=1.6",
        "rev_ai>=2.18.0",
    ],
    extras_require={
        'dev': [
            'pytest',
        ]
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities"
    ],
)


