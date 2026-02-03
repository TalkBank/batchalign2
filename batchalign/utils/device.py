import os


def apply_force_cpu() -> None:
    os.environ["BATCHALIGN_FORCE_CPU"] = "1"


def force_cpu_preferred() -> bool:
    return os.environ.get("BATCHALIGN_FORCE_CPU") == "1"
