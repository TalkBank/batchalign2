"""
dispatch.py
CLI runner dispatch. Essentially the translation layer between `command` in CLI
and actual BatchalignPipeline.
"""

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from urllib.parse import urlparse

import warnings

import shutil
import os
import glob
import shutil


from batchalign.pipelines import BatchalignPipeline
from batchalign.document import *
from batchalign.constants import *
from batchalign.formats.chat import CHATFile
from batchalign.utils import config

from rich.console import Console
from rich.markup import escape

from pathlib import Path

# Oneliner of directory-based glob and replace
globase = lambda path, statement: glob(os.path.join(path, statement))
repath_file = lambda file_path, new_dir: os.path.join(new_dir, Path(file_path).name)

import tempfile

import traceback
import logging as L 
baL = L.getLogger('batchalign')

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# this dictionary maps what commands are executed
# against what BatchalignPipeline tasks are actually ran 
Cmd2Task = {
    "align": "fa",
    "transcribe": "asr",
    "transcribe_s": "asr,speaker",
    "morphotag": "morphosyntax",
    "benchmark": "asr,eval",
    "utseg": "utterance",
    "coref": "coref",
    "translate": "translate",
}

# this is the main runner used by all functions
def _dispatch(command, lang, num_speakers,
              extensions, ctx, in_dir, out_dir,
              loader:callable, writer:callable, console,
              **kwargs):

    C = console

    # get files by walking the directory
    files = []
    outputs = []

    if kwargs.get("data"):
        url = kwargs.get("data")
        url = urlparse(url)
        if url.scheme == "":
            url = url._replace(scheme="http")
        base = os.path.basename(url.path)
        files.append(url)
        outputs.append(os.path.join(out_dir, base))

    for basedir, _, fs in os.walk(in_dir):
        for f in fs:
            path = Path(os.path.join(basedir, f))
            ext = path.suffix.strip(".").strip().lower()

            # calculate input path, convert if needed
            inp_path = str(path)
            if ext in FORCED_CONVERSION:
                # check for ffmpeg
                if not shutil.which("ffmpeg"):
                    raise ValueError(f"ffmpeg not found in Path! Cannot load input media at {inp_path}.\nHint: Please convert your input audio sample to .wav before proceeding witch Batchalign, or install ffmpeg (https://ffmpeg.org/download.html)")
                # convert
                from pydub import AudioSegment
                seg = AudioSegment.from_file(inp_path, ext)
                seg.export(inp_path.replace(f".{ext}", ".wav"), format="wav")
                inp_path = inp_path.replace(f".{ext}", ".wav")

            # repath the file to the output
            rel = os.path.relpath(inp_path, in_dir)
            repathed = Path(os.path.join(out_dir, rel))
            # make the repathed dir, if it doesn't exist
            parent = repathed.parent.absolute()
            os.makedirs(parent, exist_ok=True)

            # HACK check for @Options:\tdummy in the file
            # and simply copy it
            if ext == "cha":
                with open(inp_path, 'r', encoding="utf-8") as df:
                    data = df.read()
                if "@Options:\tdummy" in data:
                    shutil.copy2(inp_path, str(repathed))
                    continue
                elif "This is a dummy file to permit playback from the TalkBank browser" in data:
                    shutil.copy2(inp_path, str(repathed))
                    continue
                
            # if the file needs to get processed, append it to the list
            # to be processed and compute the output 
            if ext in extensions:
                files.append(inp_path)
                outputs.append(str(repathed))
            # otherwise just copy the file
            else:
                shutil.copy2(inp_path, str(repathed))

    __tf = None
    # output file
    if ctx.obj["verbose"] > 1:
        __tf = tempfile.NamedTemporaryFile(delete=True, mode='w')
        C = Console(file=__tf)

    C.print(f"\nMode: [blue]{command}[/blue]; got [bold cyan]{len(files)}[/bold cyan] transcript{'s' if len(files) > 1 else ''} to process from {in_dir}:\n")

    # create the spinner
    prog = Progress(SpinnerColumn(), *Progress.get_default_columns()[:-1],
                    TimeElapsedColumn(),
                    TextColumn("[cyan]{task.fields[processor]}[/cyan]"), console=C) 
    # cache the errors
    errors = []

    with prog as prog:
        tasks = {}
        errors = []
        # create the spinner bars
        for f in files:
            tasks[f] = prog.add_task(Path(f).name if isinstance(f, str) else Path(f.geturl()).name,
                                     start=False, processor="")

        # create pipeline and read files
        baL.debug("Attempting to create BatchalignPipeline for CLI...")
        pipeline = BatchalignPipeline.new(Cmd2Task[command],
                                          lang=lang, num_speakers=num_speakers, **kwargs)
        baL.debug(f"Successfully created BatchalignPipeline... {pipeline}")

        # create callback used to update spinner
        def progress_callback(file, step, total, tools):
            # total = 0 signals there is an error
            if total == 0:
                prog.update(tasks[file], total=0, start=True, processor=f"[bold red]FAIL[/bold red]")
            elif total == step:
                prog.update(tasks[file], total=total, completed=step, processor=f"[bold green]DONE[/bold green]")
            else:
                prog.update(tasks[file], total=total, completed=step, processor="Running: "+TaskFriendlyName[tools[0]] if tools else "")
                # call the pipeline
        for file, output in zip(files, outputs):
            try:
                # set the file as started
                prog.start_task(tasks[file])
                with warnings.catch_warnings(record=True) as w:
                    # parse the input format, as needed
                    doc = loader(os.path.abspath(file) if isinstance(file, str) else file.geturl())
                    # if we ended up with a tuple of length two,
                    # that means that the loader requested kwargs
                    kw = {}
                    if isinstance(doc, tuple) and len(doc) > 1:
                        doc, kw = doc
                    # RUN THE PUPPY!
                    doc = pipeline(doc,
                                   callback=lambda *args:progress_callback(file, *args),
                                   **kw)
                msgs = [escape(str(i.message)).strip() for i in w]
                # write the format, as needed
                writer(doc, output)
                # print any warnings
                if len(msgs) > 0:
                    if ctx.obj["verbose"] > 1:
                        Console().print(f"\n[bold yellow]WARN[/bold yellow] on {file}:\n","\n".join(msgs)+"\n")
                    else:
                        prog.console.print(f"[bold yellow]WARN[/bold yellow] on {file}:\n","\n".join(msgs)+"\n")
                prog.update(tasks[file], processor=f"[bold green]DONE[/bold green]")
            except Exception as e:
                progress_callback(file, 0, 0, e)
                errors.append((file, traceback.format_exc(), e))

    if len(errors) > 0:
        C.print()
        for file, trcbk, e in errors:
            C.print(f"[bold red]ERROR[/bold red] on file [italic]{os.path.relpath(str(Path(file).absolute()), in_dir) if isinstance(file, str) else file.geturl()}[/italic]: {escape(str(e))}\n")
            if ctx.obj["verbose"] == 1:
                C.print(escape(str(trcbk)))
            elif ctx.obj["verbose"] > 1:
                Console().print(escape(str(trcbk)))
    else:
        C.print(f"\nAll done. Results saved to {out_dir}!\n")
    if ctx.obj["verbose"] > 1:
        C.end_capture()

    if __tf:
        __tf.close()
