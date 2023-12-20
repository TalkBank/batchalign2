"""
dispatch.py
CLI runner dispatch. Essentially the translation layer between `command` in CLI
and actual BatchalignPipeline.
"""

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn

import warnings

import os
import glob

from rich.console import Console
from batchalign.pipelines import BatchalignPipeline
from batchalign.document import *
from batchalign.formats.chat import CHATFile
from batchalign.utils import config
from rich.markup import escape

# Oneliner of directory-based glob and replace
globase = lambda path, statement: glob(os.path.join(path, statement))
repath_file = lambda file_path, new_dir: os.path.join(new_dir, Path(file_path).name)

import tempfile

import traceback
import logging as L 
baL = L.getLogger('batchalign')

# this dictionary maps what commands are executed
# against what BatchalignPipeline tasks are actually ran 
Cmd2Task = {
    "align": "fa",
    "transcribe": "asr",
    "morphotag": "morphosyntax",
}

# this is the main runner used by all functions
def _dispatch(command, lang, num_speakers,
              files, ctx, in_dir, out_dir,
              loader:callable, writer:callable, console,
              **kwargs):

    C = console

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
        outputs = []
        errors = []
        # create the spinner bars
        for f in files:
            tasks[f] = prog.add_task(Path(f).name, start=False, processor="")
            outputs.append(repath_file(f, out_dir))

        # create pipeline and read files
        baL.debug("Attempting to create BatchalignPipeline for CLI...")
        pipeline = BatchalignPipeline.new(Cmd2Task[command],
                                          lang_code=lang, num_speakers=num_speakers, **kwargs)
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
                # parse the input format, as needed
                doc = loader(os.path.abspath(file))
                # RUN THE PUPPY!
                with warnings.catch_warnings(record=True) as w:
                    doc = pipeline(doc,
                                callback=lambda *args:progress_callback(file, *args))
                msgs = [str(i.message).strip() for i in w]
                # print any warnings
                if len(msgs) > 0:
                    prog.console.print(f"[bold yellow]WARN[/bold yellow] on {file}:\n","\n".join(msgs)+"\n")
                # write the format, as needed
                writer(doc, output)
                prog.update(tasks[file], processor=f"[bold green]DONE[/bold green]")
            except Exception as e:
                progress_callback(file, 0, 0, e)
                errors.append((file, traceback.format_exc(), e))

    if len(errors) > 0:
        C.print()
        for file, trcbk, e in errors:
            C.print(f"[bold red]ERROR[/bold red] on file [italic]{Path(file).name}[/italic]: {escape(str(e))}\n")
            if ctx.obj["verbose"] == 1:
                C.print(trcbk)
            elif ctx.obj["verbose"] > 1:
                Console().print(trcbk)
    else:
        C.print(f"\nAll done. Results saved to {out_dir}!\n")
    if ctx.obj["verbose"] > 1:
        C.end_capture()

    if __tf:
        __tf.close()
