# Benchmarks

Batchalign provides a lightweight benchmark command to compare feature flags on your dataset.

## Example

```bash
batchalign bench align ~/ba_data/input ~/ba_data/output --runs 3 --no-pool --no-lazy-audio --no-adaptive-workers
```

## What it measures
- Total wall-clock time per run
- Simple averages across runs

Use `--no-pool`, `--no-lazy-audio`, and `--no-adaptive-workers` to isolate performance differences.
