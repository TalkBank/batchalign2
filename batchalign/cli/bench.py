import time
import os
from pathlib import Path

import rich_click as click
from rich.console import Console

from batchalign.cli.dispatch import _dispatch


@click.command()
@click.argument("command", type=click.Choice(["align", "transcribe", "transcribe_s", "morphotag", "translate", "utseg", "benchmark", "opensmile", "coref"]))
@click.argument("in_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--runs", type=int, default=1, show_default=True, help="Number of benchmark runs.")
@click.option("--no-pool", is_flag=True, default=False, help="Disable pooled execution for this benchmark run.")
@click.option("--no-lazy-audio", is_flag=True, default=False, help="Disable lazy audio loading for this benchmark run.")
@click.option("--no-adaptive-workers", is_flag=True, default=False, help="Disable adaptive worker caps for this benchmark run.")
@click.option("--workers", type=int, default=None, help="Number of workers to use (defaults to CPU count).")
@click.pass_context
def bench(ctx, command, in_dir, out_dir, runs, no_pool, no_lazy_audio, no_adaptive_workers, workers):
    """Benchmark Batchalign command performance on a dataset."""
    console = Console()
    durations = []
    for idx in range(runs):
        run_ctx = type("Ctx", (), {"obj": dict(ctx.obj)})()
        if no_pool:
            run_ctx.obj["pool"] = False
        if no_lazy_audio:
            run_ctx.obj["lazy_audio"] = False
        if no_adaptive_workers:
            run_ctx.obj["adaptive_workers"] = False
        if workers is not None:
            run_ctx.obj["workers"] = workers
        start = time.time()
        if command in ["align", "morphotag", "translate", "utseg", "coref"]:
            extensions = ["cha"]
        elif command in ["transcribe", "transcribe_s", "benchmark", "opensmile"]:
            extensions = ["wav", "mp3", "mp4"]
        else:
            extensions = ["cha"]
        _dispatch(command, "eng", 1, extensions, run_ctx,
                  in_dir, out_dir, None, None, console)
        durations.append(time.time() - start)
        console.print(f"[dim]Run {idx+1}/{runs}:[/dim] {durations[-1]:.2f}s")
    if durations:
        avg = sum(durations) / len(durations)
        console.print(f"\n[bold]Average:[/bold] {avg:.2f}s over {len(durations)} run(s)")
