#!/usr/bin/env python3
"""
comprehensive_bench.py - Comprehensive morphotag benchmarking suite

Runs a matrix of benchmark configurations and captures detailed metrics:
- Wall-clock time
- Peak/average RSS memory
- CPU utilization
- Per-configuration statistics

Usage:
    python scripts/comprehensive_bench.py [--runs N] [--output-dir DIR]
"""

import subprocess
import time
import json
import csv
import os
import sys
import platform
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import psutil

# Configuration
BASE_INPUT = Path.home() / "ba_data" / "input-TAG"
BASE_OUTPUT = Path.home() / "ba_data" / "output-TAG"
RESULTS_DIR = Path.home() / "ba_data" / "bench-morphotag"
BATCHALIGN_DIR = Path(__file__).parent.parent


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    test_id: str
    dataset: str  # tiny, small, medium, large, full
    pool: bool
    workers: str  # "auto" or number
    adaptive: bool
    
    @property
    def input_dir(self) -> Path:
        return BASE_INPUT / self.dataset
    
    @property
    def file_count(self) -> int:
        return len(list(self.input_dir.glob("*.cha")))
    
    def to_args(self) -> list[str]:
        """Convert to batchalign bench command arguments."""
        args = ["morphotag", str(self.input_dir), str(BASE_OUTPUT)]
        if not self.pool:
            args.append("--no-pool")
        if not self.adaptive:
            args.append("--no-adaptive-workers")
        if self.workers != "auto":
            args.extend(["--workers", self.workers])
        return args
    
    def to_env(self) -> dict[str, str]:
        """Return environment variables for this config."""
        return os.environ.copy()


@dataclass
class MemorySample:
    """A single memory sample."""
    timestamp: float
    rss_mb: float
    cpu_percent: float


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    run_number: int
    start_time: str
    duration_s: float
    peak_rss_mb: float
    avg_rss_mb: float
    min_rss_mb: float
    cpu_percent_avg: float
    files_processed: int
    files_per_sec: float
    status: str  # OK, FAILED, TIMEOUT
    error_message: Optional[str] = None
    memory_samples: list[MemorySample] = field(default_factory=list)
    
    def to_csv_row(self) -> dict:
        return {
            "test_id": self.config.test_id,
            "run": self.run_number,
            "dataset": self.config.dataset,
            "files": self.files_processed,
            "pool": "yes" if self.config.pool else "no",
            "workers": self.config.workers,
            "adaptive": "yes" if self.config.adaptive else "no",
            "duration_s": f"{self.duration_s:.2f}",
            "peak_rss_mb": f"{self.peak_rss_mb:.0f}",
            "avg_rss_mb": f"{self.avg_rss_mb:.0f}",
            "cpu_pct": f"{self.cpu_percent_avg:.1f}",
            "files_per_sec": f"{self.files_per_sec:.3f}",
            "status": self.status,
        }


class MemoryMonitor:
    """Background thread that samples memory usage."""
    
    def __init__(self, pid: int, interval: float = 0.5):
        self.pid = pid
        self.interval = interval
        self.samples: list[MemorySample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> list[MemorySample]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.samples
    
    def _sample_loop(self):
        start_time = time.time()
        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        
        while not self._stop.is_set():
            try:
                # Get memory for main process and all children
                total_rss = proc.memory_info().rss
                total_cpu = proc.cpu_percent()
                
                for child in proc.children(recursive=True):
                    try:
                        total_rss += child.memory_info().rss
                        total_cpu += child.cpu_percent()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                self.samples.append(MemorySample(
                    timestamp=time.time() - start_time,
                    rss_mb=total_rss / (1024 * 1024),
                    cpu_percent=total_cpu
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            
            self._stop.wait(self.interval)


def run_benchmark(config: BenchmarkConfig, run_number: int, timeout: int = 600) -> BenchmarkResult:
    """Run a single benchmark and return results."""
    
    # Clean output directory
    for f in BASE_OUTPUT.glob("*"):
        if f.is_file():
            f.unlink()
    
    start_time = datetime.now().isoformat()
    start_ts = time.time()
    
    # Build command
    cmd = [
        sys.executable, "-m", "batchalign", "bench",
        *config.to_args(),
        "--runs", "1"
    ]
    
    print(f"  Running: {' '.join(cmd[-6:])}")
    
    try:
        # Start process
        proc = subprocess.Popen(
            cmd,
            cwd=BATCHALIGN_DIR,
            env=config.to_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Start memory monitoring
        monitor = MemoryMonitor(proc.pid)
        monitor.start()
        
        # Wait for completion
        try:
            stdout, _ = proc.communicate(timeout=timeout)
            status = "OK" if proc.returncode == 0 else "FAILED"
            error_msg = None if status == "OK" else stdout[-500:] if stdout else "Unknown error"
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            status = "TIMEOUT"
            error_msg = f"Timed out after {timeout}s"
            stdout = ""
        
        # Stop monitoring and get samples
        samples = monitor.stop()
        
    except Exception as e:
        return BenchmarkResult(
            config=config,
            run_number=run_number,
            start_time=start_time,
            duration_s=time.time() - start_ts,
            peak_rss_mb=0,
            avg_rss_mb=0,
            min_rss_mb=0,
            cpu_percent_avg=0,
            files_processed=config.file_count,
            files_per_sec=0,
            status="FAILED",
            error_message=str(e)
        )
    
    duration = time.time() - start_ts
    
    # Calculate memory stats
    if samples:
        rss_values = [s.rss_mb for s in samples]
        cpu_values = [s.cpu_percent for s in samples]
        peak_rss = max(rss_values)
        avg_rss = sum(rss_values) / len(rss_values)
        min_rss = min(rss_values)
        avg_cpu = sum(cpu_values) / len(cpu_values)
    else:
        peak_rss = avg_rss = min_rss = avg_cpu = 0
    
    files_processed = config.file_count
    files_per_sec = files_processed / duration if duration > 0 else 0
    
    return BenchmarkResult(
        config=config,
        run_number=run_number,
        start_time=start_time,
        duration_s=duration,
        peak_rss_mb=peak_rss,
        avg_rss_mb=avg_rss,
        min_rss_mb=min_rss,
        cpu_percent_avg=avg_cpu,
        files_processed=files_processed,
        files_per_sec=files_per_sec,
        status=status,
        error_message=error_msg,
        memory_samples=samples
    )


def get_system_info() -> dict:
    """Collect system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "available_memory_gb": psutil.virtual_memory().available / (1024**3),
    }


# Test matrix
TEST_CONFIGS = [
    BenchmarkConfig("T01", "tiny", pool=True, workers="auto", adaptive=True),
    BenchmarkConfig("T02", "small", pool=True, workers="auto", adaptive=True),
    BenchmarkConfig("T03", "small", pool=False, workers="auto", adaptive=True),
    BenchmarkConfig("T04", "medium", pool=True, workers="1", adaptive=False),
    BenchmarkConfig("T05", "medium", pool=True, workers="2", adaptive=False),
    BenchmarkConfig("T06", "medium", pool=True, workers="4", adaptive=False),
    BenchmarkConfig("T07", "medium", pool=True, workers="auto", adaptive=False),
    BenchmarkConfig("T08", "medium", pool=True, workers="auto", adaptive=True),
    BenchmarkConfig("T09", "large", pool=True, workers="auto", adaptive=True),
    BenchmarkConfig("T10", "large", pool=False, workers="auto", adaptive=True),
    BenchmarkConfig("T11", "full", pool=True, workers="auto", adaptive=True),
    BenchmarkConfig("T12", "full", pool=True, workers="auto", adaptive=False),
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive morphotag benchmarks")
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Results directory")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per run in seconds")
    parser.add_argument("--tests", type=str, default=None, help="Comma-separated test IDs to run (e.g., T01,T02)")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Filter tests if specified
    configs = TEST_CONFIGS
    if args.tests:
        test_ids = set(args.tests.upper().split(","))
        configs = [c for c in configs if c.test_id in test_ids]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = args.output_dir / f"bench_results_{timestamp}.csv"
    results_json = args.output_dir / f"bench_details_{timestamp}.json"
    
    print("=" * 60)
    print("COMPREHENSIVE MORPHOTAG BENCHMARK")
    print("=" * 60)
    
    system_info = get_system_info()
    print(f"\nSystem: {system_info['platform']}")
    print(f"Python: {system_info['python_version']}")
    print(f"CPUs: {system_info['cpu_count']} logical, {system_info['cpu_count_physical']} physical")
    print(f"Memory: {system_info['total_memory_gb']:.1f}GB total, {system_info['available_memory_gb']:.1f}GB available")
    
    total_runs = len(configs) * args.runs
    print(f"\nRunning {len(configs)} configs × {args.runs} runs = {total_runs} total benchmarks")
    print(f"Results will be saved to: {args.output_dir}")
    print()
    
    all_results: list[BenchmarkResult] = []
    all_details = {
        "system": system_info,
        "timestamp": timestamp,
        "runs_per_config": args.runs,
        "results": []
    }
    
    # Open CSV for writing incrementally
    csv_file = open(results_csv, "w", newline="")
    csv_writer = None
    
    try:
        run_count = 0
        for config in configs:
            print(f"\n[{config.test_id}] Dataset: {config.dataset} ({config.file_count} files)")
            print(f"        Pool: {'yes' if config.pool else 'no'}, Workers: {config.workers}, Adaptive: {'yes' if config.adaptive else 'no'}")
            
            for run in range(1, args.runs + 1):
                run_count += 1
                print(f"  Run {run}/{args.runs} ({run_count}/{total_runs} total)...", end=" ", flush=True)
                
                result = run_benchmark(config, run, timeout=args.timeout)
                all_results.append(result)
                
                # Print summary
                if result.status == "OK":
                    print(f"{result.duration_s:.1f}s | Peak: {result.peak_rss_mb:.0f}MB | {result.files_per_sec:.3f} files/s")
                else:
                    print(f"{result.status}: {result.error_message[:50] if result.error_message else 'Unknown'}")
                
                # Write to CSV
                row = result.to_csv_row()
                if csv_writer is None:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=row.keys())
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                csv_file.flush()
                
                # Add to JSON details (without memory samples to keep file size reasonable)
                detail = {
                    "test_id": config.test_id,
                    "run": run,
                    "config": asdict(config),
                    "duration_s": result.duration_s,
                    "peak_rss_mb": result.peak_rss_mb,
                    "avg_rss_mb": result.avg_rss_mb,
                    "min_rss_mb": result.min_rss_mb,
                    "cpu_percent_avg": result.cpu_percent_avg,
                    "files_processed": result.files_processed,
                    "files_per_sec": result.files_per_sec,
                    "status": result.status,
                    "memory_sample_count": len(result.memory_samples),
                }
                all_details["results"].append(detail)
    
    finally:
        csv_file.close()
    
    # Write JSON
    with open(results_json, "w") as f:
        json.dump(all_details, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Group by test_id and compute averages
    from collections import defaultdict
    by_test = defaultdict(list)
    for r in all_results:
        if r.status == "OK":
            by_test[r.config.test_id].append(r)
    
    print(f"\n{'Test':<6} {'Dataset':<8} {'Pool':<5} {'Wkrs':<5} {'Adapt':<6} {'Avg Time':<10} {'Avg Peak MB':<12} {'Files/s':<10}")
    print("-" * 75)
    
    for config in configs:
        runs = by_test.get(config.test_id, [])
        if runs:
            avg_time = sum(r.duration_s for r in runs) / len(runs)
            avg_peak = sum(r.peak_rss_mb for r in runs) / len(runs)
            avg_fps = sum(r.files_per_sec for r in runs) / len(runs)
            print(f"{config.test_id:<6} {config.dataset:<8} {'yes' if config.pool else 'no':<5} {config.workers:<5} {'yes' if config.adaptive else 'no':<6} {avg_time:>8.1f}s {avg_peak:>10.0f} {avg_fps:>9.3f}")
        else:
            print(f"{config.test_id:<6} {config.dataset:<8} {'yes' if config.pool else 'no':<5} {config.workers:<5} {'yes' if config.adaptive else 'no':<6} {'FAILED':>10}")
    
    print(f"\nResults saved to:")
    print(f"  CSV: {results_csv}")
    print(f"  JSON: {results_json}")


if __name__ == "__main__":
    main()
