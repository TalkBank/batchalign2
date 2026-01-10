"""
dispatch.py
CLI runner dispatch. Essentially the translation layer between `command` in CLI
and actual BatchalignPipeline.
"""

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn

import warnings

import shutil
import os
import glob

from rich.console import Console
from rich.markup import escape

from pathlib import Path

import concurrent.futures
import multiprocessing
from functools import partial

# Oneliner of directory-based glob and replace
globase = lambda path, statement: glob(os.path.join(path, statement))
repath_file = lambda file_path, new_dir: os.path.join(new_dir, Path(file_path).name)

import tempfile
import time

import traceback
import logging as L
baL = L.getLogger('batchalign')

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Global cache for the pipeline in worker processes
_worker_pipeline = None

def _get_worker_pipeline(command, lang, num_speakers, **kwargs):
    global _worker_pipeline
    if _worker_pipeline is None:
        from batchalign.pipelines import BatchalignPipeline
        _worker_pipeline = BatchalignPipeline.new(Cmd2Task[command],
                                                lang=lang, num_speakers=num_speakers, **kwargs)
    return _worker_pipeline

def _worker_task(file_info, command, lang, num_speakers, loader_info, writer_info, **kwargs):
    """The task executed in each worker process."""
    import sys
    import os
    import tempfile
    
    file, output = file_info
    pid = os.getpid()
    
    # Use a temporary file to capture ALL output at the FD level
    # This is the most robust way to prevent interleaved output
    with tempfile.TemporaryFile(mode='w+') as log_file:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        
        try:
            # Redirect FD 1 and 2 to our temp file
            os.dup2(log_file.fileno(), sys.stdout.fileno())
            os.dup2(log_file.fileno(), sys.stderr.fileno())
            
            pipeline = _get_worker_pipeline(command, lang, num_speakers, **kwargs)
            
            # For now, we'll re-import what we need
            from batchalign.formats.chat import CHATFile
            
            # Morphosyntax specific loader/writer logic moved here for picklability
            if command == "morphotag":
                # Extract morphotag-specific arguments from kwargs
                mwt = kwargs.pop("mwt", {})
                retokenize = kwargs.pop("retokenize", False)
                skipmultilang = kwargs.pop("skipmultilang", False)
                
                cf = CHATFile(path=os.path.abspath(file), special_mor_=True)
                doc = cf.doc
                if str(cf).count("%mor") > 0:
                    doc.ba_special_["special_mor_notation"] = True
                
                # Prepare arguments for the pipeline
                pipeline_kwargs = {
                    "retokenize": retokenize,
                    "skipmultilang": skipmultilang,
                    "mwt": mwt
                }
                # Add any remaining kwargs
                pipeline_kwargs.update(kwargs)
                
                # Process
                doc = pipeline(doc, **pipeline_kwargs)
                
                # Write
                CHATFile(doc=doc, special_mor_=doc.ba_special_.get("special_mor_notation", False)).write(output)
            
            # Add other commands as needed, or use a more generic registry
            elif command == "align":
                cf = CHATFile(path=os.path.abspath(file))
                doc = cf.doc
                kw = {"pauses": kwargs.get("pauses", False)}
                doc = pipeline(doc, **kw)
                CHATFile(doc=doc).write(output, write_wor=kwargs.get("wor", True))
            
            else:
                loader, writer = loader_info, writer_info
                doc = loader(os.path.abspath(file))
                kw = {}
                if isinstance(doc, tuple) and len(doc) > 1:
                    doc, kw = doc
                doc = pipeline(doc, **kw)
                writer(doc, output)
            
            # Flush everything before reading back
            sys.stdout.flush()
            sys.stderr.flush()
            log_file.seek(0)
            captured = log_file.read()
            
            return file, None, None, captured
        except Exception as e:
            # Flush everything before reading back
            sys.stdout.flush()
            sys.stderr.flush()
            log_file.seek(0)
            captured = log_file.read()
            return file, traceback.format_exc(), e, captured
        finally:
            # Restore original FDs
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

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
    "opensmile": "opensmile",
}

# this is the main runner used by all functions
def _dispatch(command, lang, num_speakers,
              extensions, ctx, in_dir, out_dir,
              loader:callable, writer:callable, console,
              **kwargs):

    C = console
    from batchalign.constants import FORCED_CONVERSION
    from batchalign.document import TaskFriendlyName

    # get files by walking the directory
    files = []
    outputs = []

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

    # process largest inputs first to avoid late stragglers
    file_pairs = list(zip(files, outputs))
    file_pairs.sort(key=lambda fo: os.path.getsize(fo[0]) if os.path.exists(fo[0]) else 0, reverse=True)
    files, outputs = zip(*file_pairs) if file_pairs else ([], [])

    C.print(f"\nMode: [blue]{command}[/blue]; got [bold cyan]{len(files)}[/bold cyan] transcript{'s' if len(files) > 1 else ''} to process from {in_dir}:\n")

    # Determine number of workers
    num_workers = kwargs.get("num_workers", ctx.obj.get("workers", os.cpu_count()))

    # Pre-download stanza resources if needed to avoid interleaved downloads in workers
    if command in ["morphotag", "utseg", "coref"]:
        try:
            import stanza
            stanza.download_resources_json()
        except:
            pass

    # For some commands or environments, we might want to limit this
    if command in ["transcribe", "transcribe_s"]:
        num_workers = min(num_workers, 2) # GPU memory limits

    C.print(f"Using [bold]{num_workers}[/bold] worker processes.\n")

    # create the spinner
    prog = Progress(SpinnerColumn(), *Progress.get_default_columns()[:-1],
                    TimeElapsedColumn(),
                    TextColumn("[cyan]{task.fields[processor]}[/cyan]"), console=C) 
    # cache the errors
    errors = []

    with prog as prog:
        tasks = {}
        # create the spinner bars
        for f in files:
            tasks[f] = prog.add_task(Path(f).name, start=False, processor="Waiting...")

        # Parallel execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # We pass None for loader/writer if we handle them specially in _worker_task
            worker_func = partial(_worker_task, 
                                 command=command, 
                                 lang=lang, 
                                 num_speakers=num_speakers,
                                 loader_info=None, 
                                 writer_info=None, 
                                 **kwargs)
            
            future_to_file = {executor.submit(worker_func, (f, o)): f for f, o in zip(files, outputs)}
            
            # Start all tasks in the UI to show they are queued/running
            for f in files:
                prog.start_task(tasks[f])
                prog.update(tasks[f], processor="Processing...")

            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    res_file, trcbk, e, captured = future.result()
                    if e:
                        prog.update(tasks[file], total=0, start=True, processor=f"[bold red]FAIL[/bold red]")
                        errors.append((res_file, trcbk, e, captured))
                    else:
                        prog.update(tasks[file], completed=100, total=100, processor=f"[bold green]DONE[/bold green]")
                        # If verbose, we might want to store logs even for success
                        if ctx.obj["verbose"] >= 1 and captured.strip():
                             errors.append((res_file, "Logs only (Success)", None, captured))
                except Exception as e:
                    prog.update(tasks[file], total=0, start=True, processor=f"[bold red]FAIL[/bold red]")
                    errors.append((file, traceback.format_exc(), e, ""))

    if len(errors) > 0:
        C.print()
        for file, trcbk, e, captured in errors:
            rel_path = os.path.relpath(str(Path(file).absolute()), in_dir)
            if e:
                C.print(f"[bold red]ERROR[/bold red] on file [italic]{rel_path}[/italic]: {escape(str(e))}\n")
                if captured.strip():
                    C.print(f"[dim]Captured Worker Output:[/dim]\n{escape(captured.strip())}\n")
                if ctx.obj["verbose"] == 1:
                    C.print(escape(str(trcbk)))
                elif ctx.obj["verbose"] > 1:
                    Console().print(escape(str(trcbk)))
            elif captured.strip(): # Success logs in verbose mode
                C.print(f"[bold blue]INFO[/bold blue] on file [italic]{rel_path}[/italic]:\n")
                C.print(f"{escape(captured.strip())}\n")
    else:
        C.print(f"\nAll done. Results saved to {out_dir}!\n")
    if ctx.obj["verbose"] > 1:
        C.end_capture()

    if __tf:
        __tf.close()

    if len(errors) > 0:
        C.print()
        for file, trcbk, e in errors:
            C.print(f"[bold red]ERROR[/bold red] on file [italic]{os.path.relpath(str(Path(file).absolute()), in_dir)}[/italic]: {escape(str(e))}\n")
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
