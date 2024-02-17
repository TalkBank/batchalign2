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
    name = "batchalign",
    author = "Brian MacWhinney, Houjun Liu",
    author_email = "macw@cmu.edu, houjun@cmu.edu",
    version = VERSION_NUMBER.strip(),
    description = ("Python Speech Language Sample Analysis"),
    packages=find_packages(),
    long_description=read('README.md'),
    entry_points = {
        'console_scripts': ['batchalign=batchalign.cli.cli:batchalign'],
    },
    long_description_content_type='text/markdown',
    install_requires=[
        "pydantic>=2.4",
        "nltk>=3.8",
        "praatio>=6.0.0,<6.1.0",
        "torch>=2.0.1,<2.2.0",
        "torchaudio>=2.1.0,<2.2.0",
        "pyAudioAnalysis==0.3.14",
        "hmmlearn==0.3.0",
        "eyed3~=0.9.7",
        "pydub>=0.25.1,<0.26.0",
        "imblearn",
        "plotly>=5.18.0",
        "transformers>=4.37",
        "tokenizers>=0.14.1",
        "pycountry>=22.3",
        "stanza>=1.7",
        "scipy~=1.11",
        "rev_ai>=2.18.0",
        "rich~=13.6",
        "click~=8.1",
        "matplotlib>=3.8.0,<4.0.0",
        "pyfiglet==1.0.2",
        "soundfile~=0.12.0",
        "rich-click>=1.7.0",
        "typing-extensions",
        "pydub~=0.25.0",
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
    ]
)


