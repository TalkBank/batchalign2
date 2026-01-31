# Align Memory Consumption Deep Dive & Reduction Options
Date: 2026-01-31

## Summary
`batchalign align` can consume ~3–4+ GB per worker on this machine, and scaling worker count multiplies that footprint. The dominant costs are full‑file audio tensors plus large model weights and intermediate inference buffers (forced alignment and UTR). This is partly unavoidable given current model architectures, but there are practical changes that can reduce peak memory or improve sharing.

## Current memory behavior (from code + memlog)
- Per‑worker RSS peaks observed: ~3.0–4.2 GB.
- With 10 workers on a 34 GB system, available memory dropped below 8 GB while new workers were still being scheduled.
- Low‑memory warnings appear while workers are still active, consistent with a crash when aggregate RSS exceeds RAM.

## Why memory is high (align‑specific)
### 1) Full audio tensors per worker (major)
- `WhisperFAModel`, `Wave2VecFAModel`, and `WhisperASRModel` call `torchaudio.load()` and keep a full‑length tensor in memory.
- Slices are used for alignment, but the full tensor remains resident per worker.

### 2) Model weights duplicated per process (major)
- Each worker is a separate process; models are loaded inside each worker.
- There is **no cross‑process model sharing**, so weights are duplicated N times.

### 3) Forced alignment intermediates (major)
- Whisper FA uses `output_attentions=True` and concatenates cross‑attention tensors, which can be large.
- Wave2Vec FA computes emission matrices over segments; still heavy for long spans.

### 4) Utterance timing recovery (UTR) (moderate)
- Whisper UTR loads a full ASR model and processes full audio if utterance timings are missing.
- Rev UTR avoids local model compute, but still processes full CHAT + alignment.

### 5) Document copying (moderate)
- `BatchalignPipeline` deep‑copies the document before processing; this duplicates large transcripts in memory.

### 6) Conversion overhead (avoidable)
- mp4→wav conversion previously re‑ran every time; now skipped when a same‑basename `.wav` exists.

## Are we “caching all models”?
- **Within a worker**: yes. `_worker_pipeline` is cached globally per process; engines keep their models once loaded.
- **Across workers**: no. Each process loads its own model copy. This is the primary source of multiplicative memory usage.
- **Result caching**: alignment/UTR results are cached, but this does not reduce model memory.

## What’s unavoidable vs avoidable
### Unavoidable (given current architecture)
- Model weights (Whisper/Wav2Vec) + large inference buffers are inherently large.
- Full audio tensor loads unless we switch to streaming.

### Avoidable or reducible
- **Process duplication**: sharing models across workers or reducing worker count.
- **Full‑file audio loads**: stream or map audio to avoid full‑tensor residency.
- **UTR work**: skip when utterance timings already exist.
- **Whisper FA attention buffers**: optional alternative or chunking to reduce attention tensor sizes.

## Reduction options (practical changes)
### A) Worker‑level controls (highest impact)
1) **Adaptive worker cap** based on observed RSS peaks (see proposal).
2) Default `align` worker cap (e.g., min(cpu_count, 6)).
3) Per‑run limit flag in docs (recommend <=6 on 34 GB systems).

### B) Avoid duplicate model loads
1) **Threaded inference + shared models**
   - Use a smaller number of model processes (1–2) with task queues.
   - Avoids N copies of weights.
2) **Model server process**
   - Central worker owns model; children send audio segments for alignment.

### C) Reduce per‑worker memory
1) **Stream audio**: load only needed segments from disk rather than full tensor.
2) **Shorter FA segments**: configurable max segment length reduces emission/attention size.
3) **Lower precision / quantization** where supported.
4) **Explicit GC between files**: release tensors and call `torch.cuda.empty_cache()` for GPU builds.

### D) Skip work where possible
1) **Skip UTR** when utterance timings already exist (documented flag or auto‑detect).
2) **Prefer Wave2Vec FA** over Whisper FA for memory reasons.

### E) Conversion caching
- Already addressed: skip mp4 conversion if `.wav` exists.

## Suggested next steps
1) **Implement adaptive worker cap** (proposed separately) to prevent OOM without sacrificing throughput.
2) **Add optional streaming audio path** for FA engines to avoid full‑file tensors.
3) **Add a “model‑shared” mode** (single model process + job queue) for memory‑constrained machines.
4) **Extend memlog** to record per‑file audio duration and model type to better correlate with RSS peaks.

## Immediate operational guidance
- For 34 GB systems: use `--workers 5` or `--workers 6`.
- Pre‑convert mp4 to wav (or keep wavs beside mp4s) to avoid pre‑run conversion overhead.
- Use Rev UTR (default) and Wave2Vec FA (default) to minimize memory.

## Trade-offs for single-server operation
### Option 1: Multiple manual CLI processes (historical workaround)
- **Pros**: no code changes; users can scale up or down; isolates crashes to a subset of files.
- **Cons**: completely uncoordinated memory usage; easy to exceed RAM; no shared logging, no backpressure, no centralized progress; duplicated model loads per process; difficult for non-experts to tune.
- **Net**: better than strict sequential, but unreliable and user-hostile.

### Option 2: Built-in multiprocessing with adaptive workers + mem-guard (current)
- **Pros**: centralized scheduling; adaptive cap prevents most OOM; mem-guard can fail fast; single CLI for users; predictable output; easier telemetry.
- **Cons**: each worker still loads its own models; peak RSS scales with workers; still vulnerable to very large inputs if caps are misestimated.
- **Net**: best short-term balance of throughput and safety on a single server.

### Option 3: Shared-model fork (prefork, now removed)
- **Pros**: would reduce memory by sharing read-only weights.
- **Cons**: unsafe with MPS; fragile across platforms; still duplicates audio tensors; high crash risk on macOS.
- **Net**: not viable on this hardware.

### Option 4: Model-server architecture
- **Pros**: true single-copy model weights; explicit backpressure; can batch for throughput; avoids fork/MPS issues.
- **Cons**: higher implementation complexity; new IPC bottlenecks; server crash can stall all work; serialization overhead.
- **Net**: best long-term reliability + memory profile, but slower to deliver.

### Recommendation (single server, now)
Stay with built-in multiprocessing and adaptive caps/mem-guard. It replaces the manual multi-process hack with coordinated scheduling and safety controls without requiring a complex refactor.

### New: persist memory history
We now cache the median worker RSS peaks and file-size ratios per command so adaptive caps can start with a better estimate before any workers finish. This improves the initial cap decision and reduces early over-commit on cold starts.
