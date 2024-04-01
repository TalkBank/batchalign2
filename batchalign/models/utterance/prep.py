# glob utilities
import os
import glob

# string templates
import string

# training utilities
from batchalign.models.training import *

# to run CLAN
import subprocess

import logging
L = logging.getLogger("batchalign")


# read all chat files
def read_file(f):
    """Utility to read a single flo file

    Arguments:
        f (str): the file to read

    Returns:
        list[str] a string of results
    """

    # open and read file
    with open(f, 'r') as df:
        # read!
        lines = df.readlines()

    # coallate results
    results = []

    # process lines for tab-deliminated run-on lines
    for line in lines:
        # if we have a tab
        if line[0] == '\t':
            # take away the tab, append, and put back in results
            results.append(results.pop()+" "+line.strip())
        # otherwise, just append
        else:
            results.append(line.strip())

    # return results
    return results

def prep(t):
    # get in and out dir
    _, DIR, run_name = t.resolve_data()

    if not os.path.isfile(os.path.join(DIR, f"{run_name}.train.txt")):
        # check if .txt exists
        if len(glob.glob(os.path.join(DIR, "train", "*.txt"))) == 0 and len(glob.glob(os.path.join(DIR, "train", "*.cha"))) != 0:
            # convert the cha files
            clan = get_clan()
            # run flo command
            subprocess.run([clan, "\"+t*\"", "+ca", os.path.join(DIR, "train", "*.cha")])
            # rename each file
            for f in glob.glob(os.path.join(DIR, "train", "*.flo.cex")):
                os.rename(f, f.replace(".flo.cex", ".txt"))

        # Search for all chat files
        chat_files = glob.glob(os.path.join(DIR, "train", "*.txt"))

        # prep all the files
        cleaned_files = sum([read_file(i) for i in chat_files], [])

        with open(os.path.join(DIR, f"{run_name}.train.txt"), 'w') as df:
            df.writelines([i+'\n' for i in cleaned_files])
    else:
        L.info(f"Path {os.path.join(DIR, f'{run_name}.train.txt')} exists, skipping prep...")

    if not os.path.isfile(os.path.join(DIR, f"{run_name}.val.txt")):
        # check if .txt exists
        if len(glob.glob(os.path.join(DIR, "val", "*.txt"))) == 0 and len(glob.glob(os.path.join(DIR, "val", "*.cha"))) != 0:
            # convert the cha files
            clan = get_clan()
            # run flo command
            subprocess.run([clan, "\"+t*\"", "+ca", os.path.join(DIR, "val", "*.cha")])
            # rename each file
            for f in glob.glob(os.path.join(DIR, "val", "*.flo.cex")):
                os.rename(f, f.replace(".flo.cex", ".txt"))

        # Search for all chat files
        chat_files = glob.glob(os.path.join(DIR, "val", "*.txt"))

        # prep all the files
        cleaned_files = sum([read_file(i) for i in chat_files], [])

        with open(os.path.join(DIR, f"{run_name}.val.txt"), 'w') as df:
            df.writelines([i+'\n' for i in cleaned_files])
    else:
        L.info(f"Path {os.path.join(DIR, f'{run_name}.val.txt')} exists, skipping prep...")




