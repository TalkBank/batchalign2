"""
cache.py
CLI subcommand for managing the Batchalign cache.

Provides commands to:
- Show cache statistics (--stats)
- Clear all cached data (--clear)
- Prepopulate cache from existing CHAT files (--warm)
"""

import os
from pathlib import Path

import rich_click as click
from rich.console import Console

C = Console()


def _format_bytes(count: int | None, precision: int = 2) -> str:
    """Format byte count as human-readable string."""
    if count is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    size = float(count)
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.{precision}f} {units[idx]}"


@click.group(invoke_without_command=True)
@click.option("--stats", is_flag=True, help="Show cache statistics.")
@click.option(
    "--clear",
    is_flag=True,
    help="Clear all cached data (requires confirmation)."
)
@click.pass_context
def cache(ctx, stats, clear):
    """Manage the Batchalign cache.

    The cache stores per-utterance analysis results to avoid redundant
    computation when re-processing unchanged content.

    Examples:
        batchalign cache --stats
        batchalign cache --clear
        batchalign cache warm INPUT_DIR --lang eng
    """
    # Handle --stats flag
    if stats:
        ctx.invoke(show_stats)
        return

    # Handle --clear flag
    if clear:
        ctx.invoke(clear_cache)
        return

    # If no flags and no subcommand, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cache.command("stats")
def show_stats():
    """Show cache statistics."""
    from batchalign.pipelines.cache import CacheManager

    manager = CacheManager()
    stats = manager.stats()

    C.print()
    C.print("[bold]Batchalign Cache Statistics[/bold]")
    C.print("-" * 35)
    C.print(f"[cyan]Location:[/cyan]     {stats['location']}")
    C.print(f"[cyan]Size:[/cyan]         {_format_bytes(stats['size_bytes'])}")
    C.print(f"[cyan]Entries:[/cyan]      {stats['total_entries']:,}")
    C.print()

    # Show breakdown by task
    if stats["by_task"]:
        C.print("[bold]By task:[/bold]")
        for task, count in sorted(stats["by_task"].items()):
            C.print(f"  {task}: {count:,} entries")
        C.print()

    # Show breakdown by engine version
    if stats["by_engine_version"]:
        # Get current stanza version to mark outdated entries
        try:
            import stanza
            current_stanza = stanza.__version__
        except ImportError:
            current_stanza = None

        C.print("[bold]Engine versions:[/bold]")
        for key, count in sorted(stats["by_engine_version"].items()):
            # Check if this version is outdated
            outdated = ""
            if current_stanza and "morphosyntax" in key:
                version_part = key.split()[-1] if " " in key else ""
                if version_part and version_part != current_stanza:
                    outdated = " [dim](outdated)[/dim]"
            C.print(f"  {key}: {count:,} entries{outdated}")
        C.print()


@cache.command("clear")
@click.confirmation_option(
    prompt="Are you sure you want to clear all cached data?"
)
def clear_cache():
    """Clear all cached data."""
    from batchalign.pipelines.cache import CacheManager

    manager = CacheManager()
    stats = manager.stats()
    entries_before = stats["total_entries"]

    bytes_freed = manager.clear()

    C.print()
    C.print(f"[bold green]Cache cleared.[/bold green]")
    C.print(f"  Entries removed: {entries_before:,}")
    C.print(f"  Space freed: {_format_bytes(bytes_freed)}")
    C.print()


@cache.command("warm")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--lang",
    default="eng",
    help="Language code (3-letter ISO). Default: eng"
)
@click.option(
    "--retokenize/--keeptokens",
    default=False,
    help="Whether files were processed with retokenization."
)
def warm_cache(input_dir, lang, retokenize):
    """Prepopulate cache from existing CHAT files with %mor/%gra tiers.

    Reads CHAT files that already have morphosyntactic analysis (%mor and %gra
    tiers) and populates the cache with their content. This allows subsequent
    processing of identical utterances to use cached results.

    IMPORTANT: The command trusts the input files. It does not validate that
    the %mor/%gra content is correct.
    """
    from batchalign.pipelines.cache import (
        CacheManager, MorphotagCacheKey, _get_batchalign_version
    )
    from batchalign.formats.chat import CHATFile
    from batchalign.document import Utterance

    # Get engine version
    try:
        import stanza
        engine_version = stanza.__version__
    except ImportError:
        C.print("[bold red]Error:[/bold red] stanza is not installed. Cannot warm cache.")
        return

    manager = CacheManager()
    key_gen = MorphotagCacheKey()
    ba_version = _get_batchalign_version()

    # Collect all .cha files
    cha_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".cha"):
                cha_files.append(os.path.join(root, f))

    if not cha_files:
        C.print(f"[bold yellow]No .cha files found in {input_dir}[/bold yellow]")
        return

    C.print(f"\nWarming cache from {len(cha_files)} CHAT file(s)...")
    C.print(f"  Language: {lang}")
    C.print(f"  Retokenize: {retokenize}")
    C.print(f"  Stanza version: {engine_version}")
    C.print()

    entries_added = 0
    entries_skipped = 0
    files_processed = 0

    for cha_path in cha_files:
        try:
            cf = CHATFile(path=cha_path, special_mor_=True)
            doc = cf.doc
            
            # Map for batching within a file
            utterances_to_check = []
            idx_to_key = {}
            
            for idx, item in enumerate(doc.content):
                if not isinstance(item, Utterance):
                    continue

                # Check if utterance has morphology/dependency
                has_morphology = any(
                    form.morphology and len(form.morphology) > 0
                    for form in item.content
                )
                has_dependency = any(
                    form.dependency and len(form.dependency) > 0
                    for form in item.content
                )

                if not (has_morphology or has_dependency):
                    continue

                # Generate cache key
                key = key_gen.generate_key(
                    item,
                    lang=lang,
                    retokenize=retokenize,
                    mwt={}
                )
                utterances_to_check.append((idx, key))
                idx_to_key[idx] = key

            if not utterances_to_check:
                files_processed += 1
                continue

            # Batch check
            keys = [k for _, k in utterances_to_check]
            cached_results = manager.get_batch(keys, "morphosyntax", engine_version)
            
            entries_skipped += len(cached_results)
            
            # Filter out already cached ones and prepare for batch put
            to_put = []
            for idx, key in utterances_to_check:
                if key not in cached_results:
                    item = doc.content[idx]
                    data = key_gen.serialize_output(item)
                    to_put.append((key, data))
            
            if to_put:
                manager.put_batch(to_put, "morphosyntax", engine_version, ba_version)
                entries_added += len(to_put)

            files_processed += 1

        except Exception as e:
            C.print(f"[yellow]Warning:[/yellow] Could not process {cha_path}: {e}")
            continue

    C.print(f"[bold green]Cache warming complete.[/bold green]")
    C.print(f"  Files processed: {files_processed}")
    C.print(f"  Entries added: {entries_added}")
    C.print(f"  Entries skipped (already cached): {entries_skipped}")
    C.print()
