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
import threading
import contextlib
import io

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
import json
import time

import traceback
import logging as L
baL = L.getLogger('batchalign')
import psutil
from platformdirs import user_cache_dir
from batchalign.utils.device import apply_force_cpu, force_cpu_preferred
from batchalign.pipelines import dispatch as pipeline_dispatch

POOL_UNSAFE_ENGINES = {
    "whisper",
    "whisperx",
    "whisper_oai",
    "whisper_fa",
    "wav2vec_fa",
    "whisper_utr",
    "stanza",
    "stanza_utt",
    "stanza_coref",
    "pyannote",
    "nemo_speaker",
    "seamless_translate",
    "opensmile_egemaps",
    "opensmile_gemaps",
    "opensmile_compare",
    "opensmile_eGeMAPSv01b",
}

POOL_SAFE_ENGINES = {
    "rev",
    "rev_utr",
    "evaluation",
    "gtrans",
    "replacement",
    "ngram",
}

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

def _run_pipeline_for_file(command, pipeline, file, output, loader_info, writer_info, progress_queue=None, **kwargs):
    def progress_callback(completed, total, tasks):
        if not progress_queue:
            return
        try:
            progress_queue.put((file, completed, total, tasks))
        except Exception:
            pass

    # For now, we'll re-import what we need
    from batchalign.formats.chat import CHATFile
    local_kwargs = dict(kwargs)

    # Morphosyntax specific loader/writer logic moved here for picklability
    if command == "morphotag":
        # Extract morphotag-specific arguments from kwargs
        mwt = local_kwargs.pop("mwt", {})
        retokenize = local_kwargs.pop("retokenize", False)
        skipmultilang = local_kwargs.pop("skipmultilang", False)
        override_cache = local_kwargs.pop("override_cache", False)

        cf = CHATFile(path=os.path.abspath(file), special_mor_=True)
        doc = cf.doc
        if str(cf).count("%mor") > 0:
            doc.ba_special_["special_mor_notation"] = True

        # Prepare arguments for the pipeline
        pipeline_kwargs = {
            "retokenize": retokenize,
            "skipmultilang": skipmultilang,
            "mwt": mwt,
            "override_cache": override_cache
        }
        # Add any remaining kwargs
        pipeline_kwargs.update(local_kwargs)

        # Process
        doc = pipeline(doc, callback=progress_callback, **pipeline_kwargs)

        # Write
        CHATFile(doc=doc, special_mor_=doc.ba_special_.get("special_mor_notation", False)).write(output)

    # Add other commands as needed, or use a more generic registry
    elif command == "align":
        cf = CHATFile(path=os.path.abspath(file))
        doc = cf.doc
        kw = {"pauses": local_kwargs.get("pauses", False)}
        doc = pipeline(doc, callback=progress_callback, **kw)
        CHATFile(doc=doc).write(output, write_wor=local_kwargs.get("wor", True))

    elif command in ["transcribe", "transcribe_s"]:
        from batchalign.document import CustomLine, CustomLineType
        # For transcribe, the "loader" just passes the file path
        doc = file

        # Process through pipeline
        doc = pipeline(doc, callback=progress_callback)

        # Write output with ASR comment
        asr = local_kwargs.get("asr", "rev")
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
                                write_wor=local_kwargs.get("wor", False))

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
        import os as _os
        _os.remove(P(output).with_suffix(".cha"))
        with open(P(output).with_suffix(".wer.txt"), 'w') as df:
            df.write(str(doc["wer"]))
        with open(P(output).with_suffix(".diff"), 'w') as df:
            df.write(str(doc["diff"]))
        CHATFile(doc=doc["doc"]).write(str(P(output).with_suffix(".asr.cha")),
                                       write_wor=local_kwargs.get("wor", False))

    elif command == "opensmile":
        from batchalign.document import Document
        doc = Document.new(media_path=file, lang=local_kwargs.get("lang", kwargs.get("lang", "eng")))
        results = pipeline(doc, callback=progress_callback, feature_set=local_kwargs.get("feature_set", "eGeMAPSv02"))

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

    # Always capture output to avoid interleaving with progress rendering,
    # unless high verbosity is requested for debugging.
    should_capture = verbose < 2

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
        _run_pipeline_for_file(command, pipeline, file, output, loader_info, writer_info,
                               progress_queue=progress_queue, **kwargs)

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
    worker_handled = {
        "align",
        "transcribe",
        "transcribe_s",
        "translate",
        "morphotag",
        "utseg",
        "coref",
        "benchmark",
        "opensmile",
    }
    if command in worker_handled:
        # Avoid pickling CLI-local loader/writer functions when the worker
        # implements the command-specific IO logic.
        loader = None
        writer = None
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
                wav_path = inp_path.replace(f".{ext}", ".wav")
                if not os.path.exists(wav_path):
                    # convert
                    from pydub import AudioSegment
                    seg = AudioSegment.from_file(inp_path, ext)
                    seg.export(wav_path, format="wav")
                inp_path = wav_path

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
    memlog_enabled = ctx.obj.get("memlog", False)
    mem_guard_enabled = ctx.obj.get("mem_guard", False)
    adaptive_workers_enabled = ctx.obj.get("adaptive_workers", True)
    adaptive_safety_factor = ctx.obj.get("adaptive_safety_factor", 1.35)
    adaptive_warmup = max(1, int(ctx.obj.get("adaptive_warmup", 2)))
    force_cpu_flag = ctx.obj.get("force_cpu", False)
    shared_models_requested = ctx.obj.get("shared_models", False)
    pool_requested = ctx.obj.get("pool", True)
    lazy_audio_enabled = ctx.obj.get("lazy_audio", True)
    pool_mode_enabled = False
    pool_mode_reason = None
    engine_specs = None
    memlog_path = None
    memlog_fp = None
    if memlog_enabled:
        try:
            os.makedirs(out_dir, exist_ok=True)
            memlog_path = Path(out_dir) / "batchalign_memlog.jsonl"
            memlog_fp = open(memlog_path, "a", encoding="utf-8")
        except Exception as e:
            memlog_enabled = False
            console.print(f"[bold yellow]Warning:[/bold yellow] Failed to open memlog: {e}")
    if force_cpu_flag:
        apply_force_cpu()
        C.print("[bold yellow]CPU-only mode enabled; disabling CUDA/MPS.[/bold yellow]")
    if not lazy_audio_enabled:
        from batchalign.models.utils import set_lazy_audio_enabled
        set_lazy_audio_enabled(False)
        C.print("[bold yellow]Lazy audio disabled; using full audio loads.[/bold yellow]")
    memory_history_path = Path(user_cache_dir("batchalign", "batchalign")) / "memory_history.json"
    memory_history_path.parent.mkdir(parents=True, exist_ok=True)

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

    shared_models_active = False
    mp_ctx = None
    if shared_models_requested:
        if os.name == "nt":
            C.print("[bold yellow]Shared models unavailable on Windows (spawn only).[/bold yellow]")
        elif not force_cpu_preferred() and hasattr(os, "uname") and os.uname().sysname == "Darwin":
            C.print("[bold yellow]Shared models require --force-cpu on macOS.[/bold yellow]")
        else:
            mp_ctx = multiprocessing.get_context("fork")
            shared_models_active = True
    # Determine whether pooled (threaded) execution is safe for this run.
    if pool_requested and len(files) > 1:
        try:
            engine_specs = pipeline_dispatch.resolve_engine_specs(Cmd2Task[command], lang, num_speakers, **kwargs)
        except Exception as e:
            pool_mode_reason = f"Unable to resolve engines ({e})."
        else:
            engine_names = [engine for _, engine in engine_specs]
            unsafe = [engine for engine in engine_names if engine in POOL_UNSAFE_ENGINES]
            unknown = [engine for engine in engine_names if engine not in POOL_UNSAFE_ENGINES and engine not in POOL_SAFE_ENGINES]
            if unsafe:
                pool_mode_reason = "Engines not pool-safe: " + ", ".join(sorted(set(unsafe)))
            elif unknown:
                pool_mode_reason = "Unknown pool safety: " + ", ".join(sorted(set(unknown)))
            else:
                pool_mode_enabled = True
    else:
        pool_mode_reason = None

    if pool_mode_enabled and shared_models_requested:
        C.print("[bold yellow]Shared models ignored[/bold yellow]: pooled mode uses threads.")
        shared_models_active = False
        mp_ctx = None

    if pool_mode_enabled:
        C.print(f"Using [bold]{num_workers}[/bold] worker threads (pooled models).\n")
    elif pool_mode_reason:
        if ctx.obj.get("verbose", 0) >= 1:
            C.print(f"[bold yellow]Pooled mode disabled:[/bold yellow] {pool_mode_reason} Falling back to worker processes.")
        else:
            C.print("[bold yellow]Pooled mode disabled for safety;[/bold yellow] using worker processes.")
        if shared_models_active:
            C.print(f"Using [bold]{num_workers}[/bold] worker processes (shared models).\n")
        else:
            C.print(f"Using [bold]{num_workers}[/bold] worker processes.\n")
    else:
        if shared_models_active:
            C.print(f"Using [bold]{num_workers}[/bold] worker processes (shared models).\n")
        else:
            C.print(f"Using [bold]{num_workers}[/bold] worker processes.\n")

    if pool_mode_enabled:
        manager = None
        progress_queue = queue.Queue()
    else:
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
    adaptive_cap_reported = None
    history_sizes = []
    history_peaks = []
    history_peak_est = None
    history_ratio_est = None

    def _load_memory_history():
        if not memory_history_path.exists():
            return
        try:
            payload = json.loads(memory_history_path.read_text())
        except Exception:
            return
        if payload.get("version") != 1:
            return
        command_data = payload.get("commands", {}).get(command)
        if not command_data:
            return
        peaks = command_data.get("peaks", [])
        sizes = command_data.get("sizes", [])
        if peaks:
            peaks = sorted(int(p) for p in peaks if p)
            if peaks:
                history_peaks.extend(peaks)
        if sizes:
            sizes = [int(s) for s in sizes if s]
            if sizes:
                history_sizes.extend(sizes)

    def _persist_memory_history():
        if not mem_records:
            return
        records = []
        for path, info in mem_records.items():
            peak = info.get("rss_peak") or info.get("rss_end")
            size = file_sizes.get(path, 0)
            if peak and size:
                records.append((size, peak))
        if not records:
            return
        try:
            payload = {}
            if memory_history_path.exists():
                payload = json.loads(memory_history_path.read_text())
        except Exception:
            payload = {}
        if payload.get("version") != 1:
            payload = {"version": 1, "commands": {}}
        commands = payload.setdefault("commands", {})
        data = commands.setdefault(command, {"peaks": [], "sizes": []})
        peaks = data.setdefault("peaks", [])
        sizes = data.setdefault("sizes", [])
        for size, peak in records:
            peaks.append(int(peak))
            sizes.append(int(size))
        peaks[:] = peaks[-100:]
        sizes[:] = sizes[-100:]
        try:
            memory_history_path.write_text(json.dumps(payload))
        except Exception:
            pass

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
        ratios = [mem / size for size, mem in mem_samples if size and mem]
        if not ratios and history_ratio_est:
            ratios = [history_ratio_est]
        if ratios:
            ratios.sort()
            median_ratio = ratios[len(ratios) // 2]
            est = int(median_ratio * file_size)
            return max(512 * 1024 * 1024, min(est, 6 * 1024 * 1024 * 1024))
        if history_peak_est:
            return max(512 * 1024 * 1024, min(int(history_peak_est), 6 * 1024 * 1024 * 1024))
        return 512 * 1024 * 1024

    def _should_throttle(est_bytes):
        total, available = _system_memory()
        if total is None or available is None:
            return False, total, available
        reserve = _memory_reserve(total)
        if reserve is None:
            return False, total, available
        return (available - est_bytes) < reserve, total, available

    def _adaptive_cap():
        total, available = _system_memory()
        reserve = _memory_reserve(total)
        if reserve is None or available is None:
            return num_workers
        peaks = sorted(mem for _, mem in mem_samples if mem)
        if not peaks and history_peak_est:
            peaks = [history_peak_est]
        if not peaks:
            return min(num_workers, adaptive_warmup)
        peak_est = peaks[len(peaks) // 2] * adaptive_safety_factor
        if peak_est <= 0:
            return num_workers
        cap = int((available - reserve) // peak_est)
        return max(1, min(num_workers, cap))

    def _log_event(event, **data):
        if not memlog_enabled or memlog_fp is None:
            return
        total, available = _system_memory()
        record = {
            "ts": time.time(),
            "event": event,
            "command": command,
            "total_bytes": total,
            "available_bytes": available,
            "workers": num_workers,
            "shared_models": shared_models_active,
            "pooled": pool_mode_enabled,
        }
        record.update(data)
        memlog_fp.write(json.dumps(record) + "\n")
        memlog_fp.flush()

    if not pool_mode_enabled:
        _load_memory_history()
        if history_peaks:
            history_peaks.sort()
            history_peak_est = history_peaks[len(history_peaks) // 2]
        if history_sizes and history_peaks:
            ratio_pairs = [peak / size for size, peak in zip(history_sizes, history_peaks) if size and peak]
            if ratio_pairs:
                ratio_pairs.sort()
                history_ratio_est = ratio_pairs[len(ratio_pairs) // 2]

    try:
        with prog as prog:
            if not pool_mode_enabled and history_peak_est:
                prog.console.print(
                    f"[dim]Adaptive warm start:[/dim] {_format_bytes(history_peak_est)} median peak from history"
                )
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
                    if pool_mode_enabled:
                        mem_label = "running"
                    else:
                        total_mem, available_mem = _system_memory()
                        reserve = _memory_reserve(total_mem)
                        low_mem = False
                        if reserve is not None and available_mem is not None:
                            low_mem = available_mem < reserve
                        mem_label = _mem_label("running", available_mem, low_mem)
                    prog.update(tasks[file],
                                total=task_total,
                                completed=min(int(completed), task_total),
                                processor=render_stage(stage_tasks),
                                mem=mem_label)

            if pool_mode_enabled:
                from batchalign.pipelines import BatchalignPipeline
                pipeline = BatchalignPipeline.new(Cmd2Task[command], lang=lang, num_speakers=num_speakers, **kwargs)
                shared_lock = threading.Lock()

                def _pool_worker(file_path, output_path):
                    rss_start = None
                    rss_end = None
                    rss_peak = None
                    try:
                        rss_start = psutil.Process(os.getpid()).memory_info().rss
                    except Exception:
                        rss_start = None
                    try:
                        with shared_lock:
                            _run_pipeline_for_file(command, pipeline, file_path, output_path, loader, writer,
                                                   progress_queue=progress_queue, lang=lang, **kwargs)
                        try:
                            rss_end = psutil.Process(os.getpid()).memory_info().rss
                        except Exception:
                            rss_end = None
                        mem_info = {
                            "pid": os.getpid(),
                            "rss_start": rss_start,
                            "rss_end": rss_end,
                            "rss_peak": rss_end,
                        }
                        return file_path, None, None, "", mem_info
                    except Exception as exc:
                        try:
                            rss_end = psutil.Process(os.getpid()).memory_info().rss
                        except Exception:
                            rss_end = None
                        mem_info = {
                            "pid": os.getpid(),
                            "rss_start": rss_start,
                            "rss_end": rss_end,
                            "rss_peak": rss_end,
                        }
                        return file_path, traceback.format_exc(), exc, "", mem_info

                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    future_to_file = {}
                    for file_path, output_path in zip(files, outputs):
                        future = executor.submit(_pool_worker, file_path, output_path)
                        future_to_file[future] = file_path
                        _log_event("submit_worker", file=str(file_path))
                        prog.update(tasks[file_path], processor="Processing...", mem="running")

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
                                if mem_info:
                                    mem_records[file] = mem_info
                                    peak = mem_info.get("rss_peak") or mem_info.get("rss_end")
                                    if peak:
                                        prog.update(tasks[file], mem=_format_bytes(peak))
                                    _log_event("worker_complete",
                                               file=str(file),
                                               rss_peak=mem_info.get("rss_peak"),
                                               rss_end=mem_info.get("rss_end"),
                                               rss_start=mem_info.get("rss_start"))
                            except Exception as e:
                                final_total = max(task_totals.get(file, 1), 1)
                                prog.update(tasks[file], total=final_total, completed=final_total, processor="[bold red]FAIL[/bold red]")
                                errors.append((file, traceback.format_exc(), e, ""))

                        drain_progress_queue()
            else:
                executor_opts = {}
                if mp_ctx is not None:
                    executor_opts["mp_context"] = mp_ctx
                if mp_ctx is not None and command in ["align", "morphotag"]:
                    try:
                        _get_worker_pipeline(command, lang, num_speakers, **kwargs)
                    except Exception:
                        pass

                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, **executor_opts) as executor:
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
                        _log_event("submit_worker",
                                   file=str(file_path),
                                   est_bytes=est_bytes,
                                   available_bytes=available_mem)
                        prog.update(
                            tasks[file_path],
                            processor="Processing...",
                            mem=_mem_label(f"est {_format_bytes(est_bytes)}", available_mem, low_mem),
                        )

                    def schedule_available():
                        nonlocal last_low_mem_warn, adaptive_cap_reported
                        while len(future_to_file) < num_workers:
                            if adaptive_workers_enabled:
                                cap = _adaptive_cap()
                                if cap != adaptive_cap_reported:
                                    prog.console.print(f"[dim]Adaptive cap:[/dim] {cap} worker{'s' if cap != 1 else ''}")
                                    adaptive_cap_reported = cap
                                if len(future_to_file) >= cap:
                                    break
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
                                _log_event("throttle",
                                           file=str(next_file),
                                           est_bytes=est_bytes,
                                           available_bytes=available,
                                           reserve_bytes=reserve)
                                break
                            if throttle and not future_to_file:
                                prog.console.print(
                                    f"[bold yellow]Low memory[/bold yellow]: "
                                    f"{_format_bytes(available)} free. "
                                    "Continuing with a single worker."
                                )
                                _log_event("low_mem_single_worker",
                                           file=str(next_file),
                                           est_bytes=est_bytes,
                                           available_bytes=available)
                                if mem_guard_enabled:
                                    prog.console.print(
                                        "[bold red]Memory guard[/bold red]: "
                                        "aborting to avoid system instability."
                                    )
                                    raise RuntimeError("Memory guard abort: insufficient available memory for new worker.")
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
                                    _log_event("worker_complete",
                                               file=str(file),
                                               rss_peak=mem_info.get("rss_peak"),
                                               rss_end=mem_info.get("rss_end"),
                                               rss_start=mem_info.get("rss_start"))
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
        if memlog_fp is not None:
            memlog_fp.close()
        if not pool_mode_enabled:
            _persist_memory_history()

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

    if memlog_enabled and memlog_path is not None:
        C.print(f"\n[dim]Memory log written:[/dim] {memlog_path}")

    if ctx.obj["verbose"] > 1:
        C.end_capture()

    if __tf:
        __tf.close()
