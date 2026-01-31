# Troubleshooting 

Hello! Here we collect some common troubleshooting tips when Batchalign doesn't work. **Many of these tips are for *OLDER* versions of Batchalign, and your best bet is probably to [install with the new instructions](index.md#install-and-update-the-package)** and everything will likely just work.

## Start Troubleshooting
First, to confirm your error, run Batchalign in debug mode:

```bash
batchalign -vvvv [verb] [arguments]
```

for instance:

```bash
batchalign -vvvv align ~/ba_data/input ~/ba_data/output
```

If you encounter an error, identify the **last line** of the Batchalign error output. It should usually be some coloured text placed *after* the big red box outlining the error location. It usually is shaped `[Something]Error: [error message]`. 

Once you have done this, scroll down to the matching error type to learn more.

## Tips

### Batch runs are slow or memory-heavy
Batchalign automatically uses pooled model execution when multiple files are provided, reusing models to reduce memory spikes. If pooling is unsafe for a selected engine, Batchalign falls back to process-based workers automatically. You can adjust concurrency with `--workers`.

### ConnectTimeout

```python
ConnectionTimeout: (MaxRetryError("HTTPSConnectionPool(host='[some website]')")
```

where `some website` is usually `huggingface.co` or `api.rev.ai`. 

This means we cannot reach the servers—the former, Huggingface, for model serving and the latter, Rev.AI, for ASR. Ensure those websites are accessible from the device where you are running Batchalign + also for your institution.

### `pip: command not found` or `pip3: command not found`
It is unfortunately difficult for us to provide a general command for installation that works across all systems. If your system reports either `pip` or `pip3` as not found, and you are sure *Python below 3.11 and above 3.8* is installed, try the other `pip` command. For instance, if you used `pip`, try using `pip3`.

### `batchalign: command not found` on Windows
Try using 

```
py -m batchalign [verb] [args]
```

like:

```
py -m batchalign align ~/ba_data/input ~/ba_data/output
```

### Get Pip on Windows
Run the following commands:

- `curl https://bootstrap.pypa.io/ez_setup.py | python`
- `curl https://bootstrap.pypa.io/get-pip.py | python`

Optionally, you can additionally add the path to your environment so that you can use `pip` anywhere. It's somewhere like `C:\Python33\Scripts`.
