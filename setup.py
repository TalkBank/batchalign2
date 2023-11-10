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
    version = "2.0.0",
    author = "TalkBank",
    author_email = "macw@cmu.edu",
    description = (""),
    packages=find_packages(),
    long_description=read('README.md'),
    install_requires=[
        # 'pocketsphinx',
        "pydantic>=2.4",
        "nltk>=3.8",
    ],
    extras_require={
        'dev': [
            'pytest',
        ]
    }
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities"
    ],
)


