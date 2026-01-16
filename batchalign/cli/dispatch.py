"""
dispatch.py
CLI runner dispatch. Essentially the translation layer between `command` in CLI
and actual BatchalignPipeline.
"""

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

import warnings

import shutil
import os
import glob
import queue

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
import psutil

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

def _worker_task(file_info, command, lang, num_speakers, loader_info, writer_info, progress_queue=None, verbose=0, **kwargs):
    """The task executed in each worker process."""
    import sys
    import os
    import tempfile
    import logging

    file, output = file_info
    pid = os.getpid()
    rss_start = None
    rss_end = None
    rss_peak = None

    def _safe_rss():
        try:
            import psutil
            return psutil.Process(pid).memory_info().rss
        except Exception:
            return None

    def _safe_peak_rss():
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if peak is None:
                return None
            # ru_maxrss is KB on Linux, bytes on macOS; normalize to bytes.
            return int(peak * 1024) if peak < 1024 * 1024 * 1024 else int(peak)
        except Exception:
            return None

    rss_start = _safe_rss()

    # Configure logging in this worker process
    if verbose >= 1:
        # Ensure basicConfig is called so logging works
        logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.ERROR)

    # Configure batchalign logger level in this worker process
    baL = logging.getLogger('batchalign')
    if verbose == 0:
        baL.setLevel(logging.WARN)
    elif verbose == 1:
        baL.setLevel(logging.INFO)
    else:
        baL.setLevel(logging.DEBUG)

    # Always capture output to avoid interleaving with progress rendering.
    should_capture = True

    if should_capture:
        # Use a temporary file to capture ALL output at the FD level
        # This is the most robust way to prevent interleaved output
        log_file = tempfile.TemporaryFile(mode='w+')
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())

        # Redirect FD 1 and 2 to our temp file
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())

    try:
        pipeline = _get_worker_pipeline(command, lang, num_speakers, **kwargs)

        def progress_callback(completed, total, tasks):
            if not progress_queue:
                return
            try:
                progress_queue.put((file, completed, total, tasks))
            except Exception:
                pass

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
            doc = pipeline(doc, callback=progress_callback, **pipeline_kwargs)

            # Write
            CHATFile(doc=doc, special_mor_=doc.ba_special_.get("special_mor_notation", False)).write(output)

        # Add other commands as needed, or use a more generic registry
        elif command == "align":
            cf = CHATFile(path=os.path.abspath(file))
            doc = cf.doc
            kw = {"pauses": kwargs.get("pauses", False)}
            doc = pipeline(doc, callback=progress_callback, **kw)
            CHATFile(doc=doc).write(output, write_wor=kwargs.get("wor", True))

        elif command in ["transcribe", "transcribe_s"]:
            from batchalign.document import CustomLine, CustomLineType
            # For transcribe, the "loader" just passes the file path
            doc = file

            # Process through pipeline
            doc = pipeline(doc, callback=progress_callback)

            # Write output with ASR comment
            asr = kwargs.get("asr", "rev")
            with open(Path(__file__).parent.parent / "version", 'r') as df:
                VERSION_NUMBER = df.readline().strip()
            doc.content.insert(0, CustomLine(id="Comment", type=CustomLineType.INDEPENDENT,
                                             content=f"Batchalign {VERSION_NUMBER}, ASR Engine {asr}. Unchecked output of ASR model."))
            CHATFile(doc=doc).write(output
                                    .replace(".wav", ".cha")
                                    .replace(".WAV", ".cha")
                                    .replace(".mp4", ".cha")
                                    .replace(".MP4", ".cha")
                                    .replace(".mp3", ".cha")
                                    .replace(".MP3", ".cha"),
                                    write_wor=kwargs.get("wor", False))

        elif command == "translate":
            cf = CHATFile(path=os.path.abspath(file), special_mor_=True)
            doc = cf.doc
            doc = pipeline(doc, callback=progress_callback)
            CHATFile(doc=doc).write(output)

        elif command == "utseg":
            doc = CHATFile(path=os.path.abspath(file)).doc
            doc = pipeline(doc, callback=progress_callback)
            CHATFile(doc=doc).write(output)

        elif command == "coref":
            cf = CHATFile(path=os.path.abspath(file))
            doc = cf.doc
            doc = pipeline(doc, callback=progress_callback)
            CHATFile(doc=doc).write(output)

        elif command == "benchmark":
            # Find gold transcript
            from pathlib import Path as P
            p = P(file)
            cha = p.with_suffix(".cha")
            if not cha.exists():
                raise FileNotFoundError(f"No gold .cha transcript found for benchmarking. audio: {p.name}, desired cha: {cha.name}, looked in: {str(cha)}")

            gold_doc = CHATFile(path=str(cha), special_mor_=True).doc
            doc = pipeline(file, callback=progress_callback, gold=gold_doc)

            # Write benchmark results
            import os
            os.remove(P(output).with_suffix(".cha"))
            with open(P(output).with_suffix(".wer.txt"), 'w') as df:
                df.write(str(doc["wer"]))
            with open(P(output).with_suffix(".diff"), 'w') as df:
                df.write(str(doc["diff"]))
            CHATFile(doc=doc["doc"]).write(str(P(output).with_suffix(".asr.cha")),
                                           write_wor=kwargs.get("wor", False))

        elif command == "opensmile":
            from batchalign.document import Document
            doc = Document.new(media_path=file, lang=lang)
            results = pipeline(doc, callback=progress_callback, feature_set=kwargs.get("feature_set", "eGeMAPSv02"))

            # Write opensmile results
            if results.get('success', False):
                output_csv = Path(output).with_suffix('.opensmile.csv')
                features_df = results.get('features_df')
                if features_df is not None:
                    features_df.to_csv(output_csv, header=['value'], index_label='feature')
            else:
                error_file = Path(output).with_suffix('.error.txt')
                with open(error_file, 'w') as f:
                    f.write(f"OpenSMILE extraction failed: {results.get('error', 'Unknown error')}\n")

        else:
            loader, writer = loader_info, writer_info
            if loader is None or writer is None:
                raise ValueError(f"Command '{command}' requires loader and writer functions, but they are None. This may indicate an unimplemented command or configuration issue.")
            doc = loader(os.path.abspath(file))
            kw = {}
            if isinstance(doc, tuple) and len(doc) > 1:
                doc, kw = doc
            doc = pipeline(doc, callback=progress_callback, **kw)
            writer(doc, output)

        # Flush and read captured output if we were capturing
        if should_capture:
            sys.stdout.flush()
            sys.stderr.flush()
            log_file.seek(0)
            captured = log_file.read()
        else:
            captured = ""

        rss_end = _safe_rss()
        rss_peak = _safe_peak_rss()
        mem_info = {
            "pid": pid,
            "rss_start": rss_start,
            "rss_end": rss_end,
            "rss_peak": rss_peak,
        }
        return file, None, None, captured, mem_info
    except Exception as e:
        # Flush and read captured output if we were capturing
        if should_capture:
            sys.stdout.flush()
            sys.stderr.flush()
            log_file.seek(0)
            captured = log_file.read()
        else:
            captured = ""
        rss_end = _safe_rss()
        rss_peak = _safe_peak_rss()
        mem_info = {
            "pid": pid,
            "rss_start": rss_start,
            "rss_end": rss_end,
            "rss_peak": rss_peak,
        }
        return file, traceback.format_exc(), e, captured, mem_info
    finally:
        # Restore original FDs only if we redirected them
        if should_capture:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            log_file.close()

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
    file_sizes = {f: os.path.getsize(f) if os.path.exists(f) else 0 for f in files}

    C.print(f"\nMode: [blue]{command}[/blue]; got [bold cyan]{len(files)}[/bold cyan] transcript{'s' if len(files) > 1 else ''} to process from {in_dir}:\n")

    # Determine number of workers
    num_workers = kwargs.get("num_workers", ctx.obj.get("workers", os.cpu_count()))

    # Pre-download stanza resources if needed to avoid interleaved downloads in workers
    if command in ["morphotag", "utseg", "coref"]:
        try:
            import stanza
            stanza.download_resources_json()
        except Exception:
            pass

    # For some commands or environments, we might want to limit this
    if command in ["transcribe", "transcribe_s"]:
        num_workers = min(num_workers, 2) # GPU memory limits

    C.print(f"Using [bold]{num_workers}[/bold] worker processes.\n")

    manager = multiprocessing.Manager() if files else None
    progress_queue = manager.Queue() if manager else None

    def render_stage(stage_tasks):
        if not stage_tasks:
            return "Processing..."
        if not isinstance(stage_tasks, (list, tuple)):
            stage_tasks = [stage_tasks]
        names = [TaskFriendlyName.get(task, str(task)) for task in stage_tasks]
        return ", ".join(names)

    # create the spinner
    prog = Progress(SpinnerColumn(), *Progress.get_default_columns()[:-1],
                    TimeElapsedColumn(),
                    TextColumn("[magenta]{task.fields[mem]}[/magenta]"),
                    TextColumn("[cyan]{task.fields[processor]}[/cyan]"),
                    console=C, refresh_per_second=5)
    errors = []
    mem_records = {}
    mem_samples = []
    last_low_mem_warn = 0.0

    def _format_bytes(count, precision=2):
        if count is None:
            return "unknown"
        units = ["B", "KB", "MB", "GB", "TB"]
        idx = 0
        size = float(count)
        while size >= 1024 and idx < len(units) - 1:
            size /= 1024
            idx += 1
        if idx == 0:
            return f"{int(size)}{units[idx]}"
        return f"{size:.{precision}f}{units[idx]}"

    def _mem_label(base, available=None, low_mem=False):
        parts = [base]
        if available is not None:
            parts.append(f"avail {_format_bytes(available, precision=1)}")
        if low_mem:
            parts.append("LOW MEM")
        return " | ".join(parts)

    def _system_memory():
        try:
            vm = psutil.virtual_memory()
            return vm.total, vm.available
        except Exception:
            return None, None

    def _memory_reserve(total):
        if total is None:
            return None
        return max(int(total * 0.10), 2 * 1024 * 1024 * 1024)

    def _estimate_worker_bytes(file_size):
        if not mem_samples:
            return 512 * 1024 * 1024
        ratios = [mem / size for size, mem in mem_samples if size and mem]
        if not ratios:
            return 512 * 1024 * 1024
        ratios.sort()
        median_ratio = ratios[len(ratios) // 2]
        est = int(median_ratio * file_size)
        return max(512 * 1024 * 1024, min(est, 6 * 1024 * 1024 * 1024))

    def _should_throttle(est_bytes):
        total, available = _system_memory()
        if total is None or available is None:
            return False, total, available
        reserve = _memory_reserve(total)
        if reserve is None:
            return False, total, available
        return (available - est_bytes) < reserve, total, available

    try:
        with prog as prog:
            tasks = {}
            task_totals = {}

            for f in files:
                tasks[f] = prog.add_task(Path(f).name, start=False, total=1, processor="Waiting...", mem="queued")
                task_totals[f] = 1
                prog.start_task(tasks[f])

            def drain_progress_queue():
                if not progress_queue:
                    return
                while True:
                    try:
                        file, completed, total, stage_tasks = progress_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception:
                        break
                    if file not in tasks:
                        continue
                    task_total = max(int(total) if total else task_totals.get(file, 1), 1)
                    task_totals[file] = task_total
                    total_mem, available_mem = _system_memory()
                    reserve = _memory_reserve(total_mem)
                    low_mem = False
                    if reserve is not None and available_mem is not None:
                        low_mem = available_mem < reserve
                    prog.update(tasks[file],
                                total=task_total,
                                completed=min(int(completed), task_total),
                                processor=render_stage(stage_tasks),
                                mem=_mem_label("running", available_mem, low_mem))

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                worker_func = partial(_worker_task,
                                      command=command,
                                      lang=lang,
                                      num_speakers=num_speakers,
                                      loader_info=loader,
                                      writer_info=writer,
                                      progress_queue=progress_queue,
                                      verbose=ctx.obj["verbose"],
                                      **kwargs)

                file_iter = iter(zip(files, outputs))
                future_to_file = {}

                def submit_one(file_path, output_path):
                    future = executor.submit(worker_func, (file_path, output_path))
                    future_to_file[future] = file_path
                    est_bytes = _estimate_worker_bytes(file_sizes.get(file_path, 0))
                    total_mem, available_mem = _system_memory()
                    reserve = _memory_reserve(total_mem)
                    low_mem = False
                    if reserve is not None and available_mem is not None:
                        low_mem = available_mem < reserve
                    prog.update(
                        tasks[file_path],
                        processor="Processing...",
                        mem=_mem_label(f"est {_format_bytes(est_bytes)}", available_mem, low_mem),
                    )

                def schedule_available():
                    nonlocal last_low_mem_warn
                    while len(future_to_file) < num_workers:
                        try:
                            next_file, next_output = next(file_iter)
                        except StopIteration:
                            break
                        est_bytes = _estimate_worker_bytes(file_sizes.get(next_file, 0))
                        throttle, total, available = _should_throttle(est_bytes)
                        if throttle and future_to_file:
                            now = time.time()
                            if now - last_low_mem_warn > 10:
                                reserve = _memory_reserve(total)
                                prog.console.print(
                                    f"[bold yellow]Low memory[/bold yellow]: "
                                    f"{_format_bytes(available)} free, "
                                    f"{_format_bytes(reserve)} reserve. "
                                    f"Throttling new workers."
                                )
                                last_low_mem_warn = now
                            break
                        if throttle and not future_to_file:
                            prog.console.print(
                                f"[bold yellow]Low memory[/bold yellow]: "
                                f"{_format_bytes(available)} free. "
                                "Continuing with a single worker."
                            )
                        submit_one(next_file, next_output)

                schedule_available()

                pending = set(future_to_file.keys())
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    drain_progress_queue()

                    for future in done:
                        file = future_to_file[future]
                        future_to_file.pop(future, None)
                        try:
                            res_file, trcbk, e, captured, mem_info = future.result()
                            final_total = max(task_totals.get(file, 1), 1)
                            if e:
                                prog.update(tasks[file], total=final_total, completed=final_total, processor="[bold red]FAIL[/bold red]")
                                errors.append((res_file, trcbk, e, captured))
                            else:
                                prog.update(tasks[file], total=final_total, completed=final_total, processor="[bold green]DONE[/bold green]")
                                if ctx.obj["verbose"] >= 1 and captured.strip():
                                    prog.console.print(f"[bold blue]INFO[/bold blue] on file [italic]{Path(file).name}[/italic]:\n{escape(captured.strip())}\n")
                            if mem_info:
                                mem_records[file] = mem_info
                                peak = mem_info.get("rss_peak") or mem_info.get("rss_end")
                                if peak:
                                    mem_samples.append((file_sizes.get(file, 0), peak))
                                    total_mem, available_mem = _system_memory()
                                    reserve = _memory_reserve(total_mem)
                                    low_mem = False
                                    if reserve is not None and available_mem is not None:
                                        low_mem = available_mem < reserve
                                    prog.update(tasks[file], mem=_mem_label(_format_bytes(peak), available_mem, low_mem))
                        except Exception as e:
                            final_total = max(task_totals.get(file, 1), 1)
                            prog.update(tasks[file], total=final_total, completed=final_total, processor="[bold red]FAIL[/bold red]")
                            errors.append((file, traceback.format_exc(), e, ""))

                    schedule_available()
                    pending = set(future_to_file.keys())
                drain_progress_queue()
    finally:
        if manager:
            manager.shutdown()

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
            elif captured.strip():
                C.print(f"[bold blue]INFO[/bold blue] on file [italic]{rel_path}[/italic]:\n")
                C.print(f"{escape(captured.strip())}\n")
    else:
        C.print(f"\nAll done. Results saved to {out_dir}!\n")

    if mem_records and ctx.obj["verbose"] >= 1:
        C.print("\nMemory usage per file (worker RSS peak):")
        for file, info in mem_records.items():
            rel_path = os.path.relpath(str(Path(file).absolute()), in_dir)
            peak = info.get("rss_peak") or info.get("rss_end")
            C.print(f"- {rel_path}: {_format_bytes(peak)}")
        total, available = _system_memory()
        if total is not None and available is not None:
            C.print(f"\nSystem memory available: {_format_bytes(available)} / {_format_bytes(total)}")

    if ctx.obj["verbose"] > 1:
        C.end_capture()

    if __tf:
        __tf.close()
