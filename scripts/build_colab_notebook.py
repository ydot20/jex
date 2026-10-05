#!/usr/bin/env python3
"""Build the self-contained notebook from scripts/colab_full_run.py.

Run from any directory; all paths are relative to this repository.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "colab_full_run.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "jex_gpu_full.ipynb"

INTRO = """# jex: complete configured Colab GPU run

Select a GPU runtime, then **Run all**. This notebook uses public, pinned clones, with no GitHub token, Google Drive mount, or paid model API. Runtime storage is temporary: download both output ZIPs before disconnecting.

Scope matches the original `jex_gpu.ipynb`: Qwen2.5-1.5B student, 4-bit Qwen2.5-7B teacher; **400/50/150/400** train/dev/eval/held-out records per task; **five head variants**; **3,000-record LoRA**; **four benchmark models at 30 examples per dataset**; latency K=1/2/4/8/16 with `--repeats 5`; async rates 5/20/50 with 200 arrivals each. No automatic downscaling. The 30-example benchmark is the original configured sample, **not the full benchmark dataset evaluation**. A full-dataset extension is separate and is not launched here.

Subprocess failures are visible and logged, and block dependent stages. Independent stages continue. The final status is incomplete if a stage failed or was blocked. Rerun in the same runtime to retain completed stages and retry unsuccessful ones. Package versions, source commits, commands, statuses, timings, coverage, and artifact hashes are retained. Exact training/evaluation records are saved; unpinned upstream Hub revisions and GPU nondeterminism still limit reproducibility.

**Resources:** original T4 LoRA uses float32, batch four, without gradient checkpointing. It may run out of memory; this runner reports failure rather than shrinking the experiment. Total runtime and resource usage depend on the GPU and have not been established by notebook preparation. Dataset gates and network failures are recorded in logs and coverage.

**Review the released response-data license before enabling Jev comparison.** `JEV_RESPONSE_LICENSE_ACK` defaults to `False`, so Jev download/scoring/comparison remain explicitly blocked while independent work runs. Set it to `True` only if the intended use complies with [the response-data license](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking/blob/6bbdeb33474849b6de2f0cccc9f5e19756abd67e/responses/LICENSE_RESPONSES.md). The acknowledgment does not waive restrictions. Jev responses are used only for research comparison, never for training, distillation, targets, or optimization. Their license also prohibits developing or facilitating a similar or competing product. Qwen3.8 and Gemma responses use Apache 2.0. Exports include unchanged licenses and attribution; source response databases are excluded.

Reference: Deußer, Sparrenberg, Sifa, *Evaluating and Benchmarking the System One Model Jev*, arXiv:2609.37647, dataset DOI [10.5281/zenodo.23039006](https://doi.org/10.5281/zenodo.23039006).

Pinned jex: `c494049aa3af480abc5ec502a10a783b26c2a71b`. Pinned harness: `6bbdeb33474849b6de2f0cccc9f5e19756abd67e`. See `docs/COLAB_FULL_RUN.md` for outputs, caveats, and recovery.
"""

RUN_CELL = """import subprocess, sys
command = [sys.executable, "-u", str(RUNNER), "--root", str(RUN_ROOT)]
if JEV_RESPONSE_LICENSE_ACK:
    command.append("--jev-license-ack")
# Inherit stdout/stderr so complete live logs remain in notebook output.
completed = subprocess.run(command)
print("Runner exit code:", completed.returncode)
if completed.returncode:
    print("INCOMPLETE. Check stage_status.json and logs; successful independent stages are retained.")
else:
    print("All declared stages succeeded. Inspect coverage before interpreting benchmark means.")
"""

DOWNLOAD_CELL = """import json
from google.colab import files
status_path = RUN_ROOT / "stage_status.json"
if status_path.exists():
    status = json.loads(status_path.read_text())
    for stage, item in status.items():
        print(stage, item["status"], item.get("error", item.get("reason", "")))
for name in ["results.zip", "checkpoints.zip"]:
    path = RUN_ROOT / name
    if path.exists():
        print("Downloading", path.name, f"({path.stat().st_size / 2**20:.1f} MiB)")
        files.download(str(path))
    else:
        print("Unavailable:", path)
"""


def make_cell(kind: str, text: str, index: int) -> dict:
    cell = {
        "cell_type": kind,
        "id": f"jex-full-{index}",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build_notebook(runner_source: str) -> dict:
    digest = hashlib.sha256(runner_source.encode()).hexdigest()
    bootstrap = (
        "from pathlib import Path\nimport hashlib\n\n"
        'RUN_ROOT = Path("/content/jex-full-run")\n'
        "# Review the response-data license and intended use before enabling.\n"
        "JEV_RESPONSE_LICENSE_ACK = False\n"
        'RUNNER = Path("/content/colab_full_run.py")\n'
        f"RUNNER_SOURCE = {runner_source!r}\n"
        "RUNNER.write_text(RUNNER_SOURCE)\n"
        f"assert hashlib.sha256(RUNNER.read_bytes()).hexdigest() == {digest!r}\n"
        'print("Prepared", RUNNER, "at", hashlib.sha256(RUNNER.read_bytes()).hexdigest())\n'
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": NOTEBOOK_PATH.name, "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            make_cell("markdown", INTRO, 0),
            make_cell("code", bootstrap, 1),
            make_cell("code", RUN_CELL, 2),
            make_cell("code", DOWNLOAD_CELL, 3),
        ],
    }


def main() -> None:
    notebook = build_notebook(RUNNER_PATH.read_text())
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {NOTEBOOK_PATH.relative_to(ROOT)} ({NOTEBOOK_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
