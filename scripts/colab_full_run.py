#!/usr/bin/env python3
"""Full, resumable GPU run of the pinned jex notebook. No credentials or Drive mount.

Only stdlib is imported in the orchestration process. Training, tests, evaluation,
and GPU benchmarks run in fresh subprocesses after dependencies are installed.
Run: python scripts/colab_full_run.py --root /content/jex-full-run
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as package_metadata, util as import_util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile

JEX_URL = "https://github.com/dffdeeq/jex.git"
JEX_COMMIT = "c494049aa3af480abc5ec502a10a783b26c2a71b"
HARNESS_URL = "https://github.com/AppliedMachineLearning-Lab/jev-benchmarking.git"
HARNESS_COMMIT = "6bbdeb33474849b6de2f0cccc9f5e19756abd67e"
LICENSE_URL = f"https://github.com/AppliedMachineLearning-Lab/jev-benchmarking/blob/{HARNESS_COMMIT}/responses/LICENSE_RESPONSES.md"
ATTRIBUTION = """Benchmark code and released response-data reference:
Tobias Deußer, Lorenz Sparrenberg and Rafet Sifa (2026).
Evaluating and Benchmarking the System One Model Jev. arXiv:2609.37647.
Paper: https://arxiv.org/abs/2609.37647
Response dataset: https://doi.org/10.5281/zenodo.23039006

Jev response data is for licensed research/evaluation use only; no training,
distillation, optimization, or competing-product development. Open-model
response data is separately licensed under Apache 2.0. Consult the unchanged
response-data license included with results when the harness is available.
Source response databases are not included in this archive.
"""
STUDENT = "Qwen/Qwen2.5-1.5B-Instruct"
TEACHER = "Qwen/Qwen2.5-7B-Instruct"
SIZES = {"train_per_task": 400, "dev_per_task": 50, "eval_per_task": 150, "heldout_per_task": 400}
HEADS = {
    "jex-head": [],
    "ablation-gold-only": ["--no-teacher"],
    "ablation-no-rlcd": ["--rl-epochs", "0"],
    "ablation-no-memory": ["--no-memory"],
    "prior-kl-1.0": ["--no-memory", "--prior-kl", "1.0"],
}
BENCH_MODELS = {
    "qwen2.5-1.5b-zs": (STUDENT, None),
    "jex-lora": ("artifacts/lora", None),
    "qwen3.5-4b-zs": ("Qwen/Qwen3.5-4B", None),
    "qwen3.5-9b-zs": ("Qwen/Qwen3.5-9B", "4bit"),
}
BASELINES = {
    "Qwen3.8-27B": ("hf:Qwen/Qwen3.8-27B", "responses.Qwen__Qwen3.8-27B.db"),
    "Gemma-4-E4B": ("hf:google/gemma-4-E4B-it", "responses.google__gemma-4-E4B-it.db"),
    "Jev": ("jev-1.13.0", "responses.db"),
}
BENCH_LIMIT = 30
# Preserve the Colab CUDA PyTorch build. Other resolved versions are captured by
# pip freeze and the environment manifest, rather than upgrading torch blindly.
PINNED_EXTRA_PACKAGES = ["transformers==5.14.1", "peft==0.21.2", "flash-linear-attention==0.5.2"]
CONFIG = {
    "jex_commit": JEX_COMMIT, "harness_commit": HARNESS_COMMIT,
    "student": STUDENT, "teacher": TEACHER, "teacher_quant": "4bit",
    "sizes": SIZES, "heads": HEADS, "lora_max_records": 3000,
    "lora_epochs": 1, "lora_batch_records": 4,
    "bench_models": BENCH_MODELS, "bench_limit": BENCH_LIMIT,
    "latency_ks": [1, 2, 4, 8, 16], "latency_repeats": 5,
    "async_rates": [5, 20, 50], "async_n": 200, "seed": 0,
    "pinned_extra_packages": PINNED_EXTRA_PACKAGES,
}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_version(name):
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


def torchao_is_incompatible(version):
    """Conservatively classify stable release metadata against PEFT 0.21.2.

    Prerelease or unrecognized versions are not grounds for automatic removal.
    No ML package needs to be imported for this check.
    """
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\+[A-Za-z0-9.-]+)?", version)
    if match is None:
        return None
    return tuple(int(value or 0) for value in match.groups()) < (0, 16, 0)


class Blocked(RuntimeError):
    pass


class Runner:
    def __init__(self, root, *, jev_license_ack=False, allow_colab_optional_package_cleanup=False):
        self.root = Path(root).resolve()
        self.repo = self.root / "jex"
        self.harness = self.root / "jev-benchmarking"
        self.logs = self.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.jev_license_ack = jev_license_ack
        self.allow_colab_optional_package_cleanup = allow_colab_optional_package_cleanup
        self.status_path = self.root / "stage_status.json"
        self.status = json.loads(self.status_path.read_text()) if self.status_path.exists() else {}
        config_path = self.root / "run_config.json"
        expected = json.loads(json.dumps(CONFIG))
        if config_path.exists() and json.loads(config_path.read_text()) != expected:
            raise RuntimeError("Existing run configuration differs. Use a NEW root directory; do not mix runs.")
        write_json(config_path, CONFIG)
        (self.root / "ATTRIBUTION.txt").write_text(ATTRIBUTION)
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONHASHSEED="0",
                        TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1",
                        GIT_TERMINAL_PROMPT="0")
        # These are offline/cached evaluations, not live TypeSafe API calls.
        self.env.pop("TYPESAFE_API_KEY", None)
        self.env.pop("TYPESAFE_API_KEY_FILE", None)
        self.current = None
        self.logfile = None
        self.plan = []

    def emit(self, line):
        print(line, end="" if line.endswith("\n") else "\n", flush=True)
        if self.logfile:
            self.logfile.write(line if line.endswith("\n") else line + "\n")
            self.logfile.flush()

    def command(self, args, *, cwd=None, stdout_file=None):
        args = [str(a) for a in args]
        self.emit("$ " + shlex.join(args))
        proc = subprocess.Popen(args, cwd=cwd or self.repo, env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        capture = Path(stdout_file).open("w") if stdout_file else None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.emit(line)
                if capture:
                    capture.write(line)
            code = proc.wait()
            if code:
                raise subprocess.CalledProcessError(code, args)
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if capture:
                capture.close()

    def output(self, args, cwd=None):
        return subprocess.check_output(args, cwd=cwd or self.repo, env=self.env, text=True).strip()

    def python(self, script, *args, cwd=None):
        self.command([sys.executable, "-u", self.root / "seeded_run.py", script, *args], cwd=cwd)

    def clone(self, url, commit, destination):
        if not destination.exists():
            self.command(["git", "clone", "--no-checkout", url, destination], cwd=self.root)
        actual_url = self.output(["git", "remote", "get-url", "origin"], cwd=destination)
        if actual_url != url:
            raise RuntimeError(f"Refusing unrelated existing checkout: {destination}")
        if not (destination / ".git").exists():
            raise RuntimeError(f"Not a git repository: {destination}")
        self.command(["git", "checkout", "--detach", commit], cwd=destination)
        actual = self.output(["git", "rev-parse", "HEAD"], cwd=destination)
        if actual != commit:
            raise RuntimeError(f"Revision mismatch: {actual}")
        self.emit(f"Verified {url} @ {actual}")

    def add(self, name, fn, deps=(), outputs=(), *, always=False):
        self.plan.append((name, fn, tuple(deps), [Path(p) for p in outputs], always))

    def stage(self, name, fn, deps, outputs, always):
        blockers = [d for d in deps if self.status.get(d, {}).get("status") != "succeeded"]
        if blockers:
            self.status[name] = {"status": "blocked", "reason": "Dependencies not succeeded: " + ", ".join(blockers), "updated": utc()}
            write_json(self.status_path, self.status)
            self.emit(f"BLOCKED {name}: {', '.join(blockers)}")
            return
        previous = self.status.get(name, {})
        if (not always and outputs and previous.get("status") == "succeeded"
                and all(p.is_file() and p.stat().st_size > 0 for p in outputs)):
            self.emit(f"RESUME {name}: completed stage retained")
            return
        row = {"status": "running", "started": utc(), "attempt": previous.get("attempt", 0) + 1,
               "outputs": [str(p.relative_to(self.root)) for p in outputs]}
        self.status[name] = row
        write_json(self.status_path, self.status)
        t0 = time.monotonic()
        self.current = name
        with (self.logs / f"{name}.log").open("a") as log:
            self.logfile = log
            self.emit(f"\n===== {name} [{utc()}] =====")
            try:
                fn()
                missing = [str(p) for p in outputs if not p.is_file() or p.stat().st_size == 0]
                if missing:
                    raise RuntimeError("Missing/empty expected outputs: " + ", ".join(missing))
                row["status"] = "succeeded"
            except Blocked as exc:
                row.update(status="blocked", reason=str(exc))
                self.emit(f"BLOCKED: {exc}")
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                self.emit(traceback.format_exc())
            except BaseException:
                row.update(status="interrupted", error="Notebook interrupted; rerun to resume")
                raise
            finally:
                row.update(finished=utc(), seconds=time.monotonic() - t0)
                write_json(self.status_path, self.status)
                self.logfile = None
                self.current = None
                self.bundle()
        self.emit(f"{row['status'].upper()} {name} ({row['seconds']:.1f}s)")

    def colab_runtime_evidence(self):
        try:
            has_colab_package = import_util.find_spec("google.colab") is not None
        except (ImportError, ValueError):
            has_colab_package = False
        return {
            "colab_package_available": has_colab_package,
            "colab_release_tag_present": bool(os.environ.get("COLAB_RELEASE_TAG")),
            "content_directory_present": Path("/content").is_dir(),
        }

    def prepare_optional_backends(self):
        """Remove only an unused incompatible TorchAO, with explicit Colab opt-in."""
        names = ("torch", "peft", "transformers", "torchao")
        before = {name: installed_version(name) for name in names}
        if before["torchao"] is None:
            self.emit("Optional TorchAO is absent; no compatibility cleanup needed.")
            return
        if before["peft"] != "0.21.2":
            raise RuntimeError("Unexpected PEFT version; refusing automatic optional-backend changes")
        incompatible = torchao_is_incompatible(before["torchao"])
        if incompatible is False:
            self.emit(f"Optional TorchAO {before['torchao']} meets PEFT 0.21.2's >=0.16.0 requirement; unchanged.")
            return
        evidence = self.colab_runtime_evidence()
        reason = (f"Unused TorchAO {before['torchao']} is incompatible with pinned PEFT 0.21.2 "
                  "(requires >=0.16.0 when TorchAO is installed). jex uses ordinary LoRA and bitsandbytes, not TorchAO.")
        if incompatible is None:
            reason = f"Cannot safely classify installed TorchAO {before['torchao']!r} against PEFT 0.21.2's >=0.16.0 requirement."
        record = {"checked_at": utc(), "before": before, "reason": reason,
                  "cleanup_explicitly_enabled": self.allow_colab_optional_package_cleanup,
                  "runtime_evidence": evidence, "status": "blocked"}
        path = self.root / "optional_backend_cleanup.json"
        write_json(path, record)
        self.emit("Optional-backend check: " + json.dumps(record, sort_keys=True))
        remedy = ("No package was removed. Use a fresh isolated virtual environment without --system-site-packages "
                  "and install the pinned dependencies there. In a hosted ephemeral Colab runtime only, "
                  "review and enable --allow-colab-optional-package-cleanup.")
        if incompatible is None:
            raise RuntimeError(f"Cannot safely classify TorchAO version {before['torchao']!r}. " + remedy)
        if not self.allow_colab_optional_package_cleanup or not all(evidence.values()):
            raise RuntimeError(reason + " Automatic removal requires explicit opt-in and verified hosted Colab indicators. " + remedy)
        self.emit(reason + " Removing only this unused optional package from the ephemeral Colab runtime.")
        record["status"] = "removing"
        write_json(path, record)
        try:
            self.command([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], cwd=self.root)
            after = {name: installed_version(name) for name in names}
            record["after"] = after
            if after["torchao"] is not None or any(after[name] != before[name] for name in names[:-1]):
                raise RuntimeError("Optional-backend cleanup verification failed; core package versions must remain unchanged")
            record["status"] = "removed"
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(path, record)
        self.emit("Verified optional TorchAO removal; torch, PEFT and Transformers versions are unchanged.")

    def setup(self):
        self.clone(JEX_URL, JEX_COMMIT, self.repo)
        self.clone(HARNESS_URL, HARNESS_COMMIT, self.harness)
        self.command([sys.executable, "-m", "pip", "install", "-e", f"{self.repo}[server,dev,gpu]",
                      "-e", str(self.harness), *PINNED_EXTRA_PACKAGES[:2]], cwd=self.root)
        self.prepare_optional_backends()
        (self.repo / "artifacts").mkdir(exist_ok=True)
        # The upstream harness ships published full-sample results. Preserve them
        # elsewhere, then score into an empty directory to avoid stale comparisons.
        results = self.harness / "results"
        archive = self.harness / "published-results-reference-only"
        if results.exists() and not archive.exists():
            results.rename(archive)
        results.mkdir(exist_ok=True)
        (self.harness / "cache").mkdir(exist_ok=True)
        (self.root / "seeded_run.py").write_text('''import random, runpy, sys\nimport numpy as np\nimport torch\nrandom.seed(0)\nnp.random.seed(0)\ntorch.manual_seed(0)\nif torch.cuda.is_available(): torch.cuda.manual_seed_all(0)\nscript = sys.argv[1]\nsys.argv = sys.argv[1:]\nsys.path.insert(0, str(__import__('pathlib').Path(script).resolve().parent))\nrunpy.run_path(script, run_name="__main__")\n''')
        self.command([sys.executable, "-m", "pip", "freeze"], cwd=self.root, stdout_file=self.root / "pip-freeze.txt")
        self.command(["nvidia-smi"], cwd=self.root, stdout_file=self.root / "nvidia-smi.txt")
        self.command([sys.executable, "-c", "import torch, peft, transformers, datasets, bitsandbytes, accelerate; "
            "assert torch.cuda.is_available(), 'GPU required; no CPU experiment fallback'; "
            "print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__,"
            "'peft',peft.__version__,'datasets',datasets.__version__); "
            "print('GPU',torch.cuda.get_device_name(0),'VRAM',torch.cuda.get_device_properties(0).total_memory,'bf16',torch.cuda.is_bf16_supported())"],
            cwd=self.root, stdout_file=self.root / "environment.txt")
        shutil.copy2(Path(__file__), self.root / "runner-source.py")
        self.emit("LoRA retains upstream dtype/batch defaults. PyTorch may report emulated BF16 support on T4; memory use remains unverified. No automatic sample/batch reduction.")

    def download_baseline(self, label):
        model, filename = BASELINES[label]
        if label == "Jev" and not self.jev_license_ack:
            raise Blocked("Jev response-data license review is required before use: " + LICENSE_URL)
        path = self.harness / "cache" / filename
        if not path.exists():
            url = f"https://zenodo.org/api/records/23039006/files/{filename}/content"
            partial = path.with_suffix(".part")
            self.emit("Downloading " + url)
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as target:
                shutil.copyfileobj(response, target, length=8 * 1024 * 1024)
            partial.replace(path)
        with path.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                raise RuntimeError(f"Not a SQLite response database: {path}")
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            count = conn.execute("SELECT COUNT(*) FROM responses WHERE model=?", (model,)).fetchone()[0]
            if count == 0:
                raise RuntimeError(f"Database has no responses for {model}")
        write_json(self.root / f"baseline-{label}.json", {"url": f"https://zenodo.org/api/records/23039006/files/{filename}/content",
            "filename": filename, "model": model, "responses": count, "sha256": sha256(path), "license": LICENSE_URL})

    def results_dir(self, model):
        base = self.harness / "results" / "eval"
        return base if model == "jev-1.13.0" else base / "open_models" / model.removeprefix("hf:").replace("/", "__")

    def score(self, model):
        directory = self.results_dir(model)
        directory.mkdir(parents=True, exist_ok=True)
        # A partial/failed earlier attempt must not leave stale per-task metrics.
        for old in directory.glob("*.json"):
            old.unlink()
        summary = directory / "summary.md"
        if summary.exists():
            summary.unlink()
        self.python("scripts/evaluate.py", "all", "--split", "eval", "--limit", BENCH_LIMIT,
                    "--no-ci", "--model", model, cwd=self.harness)
        rows = []
        for path in sorted(directory.glob("*.json")):
            item = json.loads(path.read_text())
            if isinstance(item, dict) and "n_examples" in item:
                rows.append({k: item.get(k) for k in ["task", "n_examples", "n_answered", "models", "primary", "skipped_configs"]})
        if not rows:
            raise RuntimeError(f"No benchmark tasks scored for {model}; consult logs, do not treat this as success")
        write_json(self.root / "coverage" / (model.replace("/", "__").replace(":", "_") + ".json"),
                   {"model": model, "requested_limit": BENCH_LIMIT, "scored_tasks": len(rows), "tasks": rows,
                    "caveat": "Dataset/config load failures and schema rejections are preserved in logs. A subprocess exit 0 is not full dataset coverage."})

    def compare(self, include_jev):
        models = (["Jev=jev-1.13.0"] if include_jev else []) + [f"{k}={BASELINES[k][0]}" for k in ("Qwen3.8-27B", "Gemma-4-E4B")]
        models += [f"{tag}=hf:jex/{tag}" for tag in BENCH_MODELS]
        # Upstream compare_jev divides by zero on no common tasks. Fail clearly.
        tables = []
        for item in models:
            _, model = item.split("=", 1)
            tasks = set()
            for file in self.results_dir(model).glob("*.json"):
                data = json.loads(file.read_text())
                if isinstance(data, dict) and (data.get("primary") or {}).get("value") is not None:
                    tasks.add(file.stem)
            tables.append(tasks)
        common = set.intersection(*tables)
        if not common:
            raise RuntimeError("No common scored benchmark tasks; cannot compute a comparison mean")
        if not (common - {"sst2", "ag_news", "emotion", "sst5"}):
            raise RuntimeError("No common untrained tasks; upstream comparison would divide by zero")
        args = ["scripts/compare_jev.py", "--harness", str(self.harness)]
        for item in models:
            args.extend(["--model", item])
        args.extend(["--out", "artifacts/jev_compare.md" if include_jev else "artifacts/open_models_compare.md"])
        self.python(*args)
        write_json(self.root / ("comparison_coverage.json" if include_jev else "open_comparison_coverage.json"),
                   {"limit": BENCH_LIMIT, "common_tasks": sorted(common),
                    "models": {name: {"scored_tasks": sorted(tasks), "excluded_from_common": sorted(tasks - common)} for name, tasks in zip(models, tables)},
                    "warning": "Aggregate uses common tasks only. Consult each model coverage file and per-task n_answered before interpreting it."})

    def make_plan(self):
        art = self.repo / "artifacts"
        self.add("setup", self.setup, always=True)
        self.add("tests", lambda: self.command([sys.executable, "-m", "pytest", "-q", "--junitxml", str(self.root / "tests.xml")]),
                 ("setup",), [self.root / "tests.xml"])
        sizes = [part for key, value in SIZES.items() for part in ["--" + key.replace("_", "-"), str(value)]]
        def extract():
            # Keep the exact original records across a same-runtime retry.
            saved = art / "feats" / "records.pt"
            resume = ["--records-from", str(saved)] if saved.exists() else []
            self.python("scripts/extract.py", "--out", "artifacts/feats", "--student", STUDENT,
                        "--teacher", TEACHER, "--teacher-quant", "4bit", "--device", "cuda", *sizes, *resume)
        self.add("extract", extract, ("tests",),
                 [art / "feats" / f"{kind}_{split}.pt" for kind in ["student", "teacher"] for split in ["train", "dev", "eval"]]
                 + [art / "feats" / "records.pt", art / "feats" / "meta.json"])
        for name, flags in HEADS.items():
            self.add("head-" + name, lambda name=name, flags=flags: self.python("scripts/train_head.py", "--out", f"artifacts/{name}", "--device", "cuda", *flags),
                     ("extract",), [art / name / "head.pt", art / name / "train_info.json", art / name / "jex.json"])
        self.add("evaluate-heads", lambda: self.python("scripts/evaluate.py", "--heads", *[f"artifacts/{name}" for name in HEADS]),
                 ["head-" + name for name in HEADS], [art / "eval.json"])
        self.add("train-lora", lambda: self.python("scripts/train_lora.py", "--out", "artifacts/lora", "--max-records", "3000", "--device", "cuda"),
                 ("extract",), [art / "lora" / f for f in ["adapter_model.safetensors", "adapter_config.json", "jex.json", "train_info.json"]])
        self.add("extract-lora", lambda: self.python("scripts/extract.py", "--student", STUDENT, "--student-adapter", "artifacts/lora", "--records-from", "artifacts/feats/records.pt",
                 "--splits", "dev,eval", "--skip-teacher", "--out", "artifacts/feats-lora", "--device", "cuda"), ("train-lora",),
                 [art / "feats-lora" / f for f in ["student_dev.pt", "student_eval.pt", "records.pt", "meta.json"]])
        self.add("evaluate-lora", lambda: self.python("scripts/evaluate.py", "--feats", "artifacts/feats-lora", "--heads", "--out", "artifacts/eval_lora.json"),
                 ("extract-lora",), [art / "eval_lora.json"])
        def install_fla():
            # Tiny Qwen3.5 pytest fixtures live on CPU. Install GPU FLA only after
            # those tests, matching the original notebook's ordering.
            self.command([sys.executable, "-m", "pip", "install", PINNED_EXTRA_PACKAGES[2]], cwd=self.root)
            self.command([sys.executable, "-m", "pip", "freeze"], cwd=self.root, stdout_file=self.root / "pip-freeze.txt")
            (self.root / "fla-installed.txt").write_text(PINNED_EXTRA_PACKAGES[2] + "\n")
        self.add("install-fla", install_fla, ("tests",), [self.root / "fla-installed.txt"])
        for tag, (model, quant) in BENCH_MODELS.items():
            deps = ("tests", "train-lora") if tag == "jex-lora" else ("tests",)
            if tag.startswith("qwen3.5"):
                deps += ("install-fla",)
            args = ["scripts/jev_bench.py", "--checkpoint" if tag == "jex-lora" else "--backbone", model,
                    "--tag", tag, "--limit", str(BENCH_LIMIT), "--out", f"runs/{tag}", "--device", "cuda"]
            if quant:
                args += ["--quant", quant]
            self.add("bench-" + tag, lambda args=args: self.python(*args), deps, [self.repo / "runs" / tag / "responses.jsonl"])
            def import_and_score(tag=tag):
                self.python("scripts/import_responses.py", str(self.repo / "runs" / tag / "responses.jsonl"), cwd=self.harness)
                self.score("hf:jex/" + tag)
            self.add("score-" + tag, import_and_score, ("bench-" + tag,), [self.results_dir("hf:jex/" + tag) / "summary.md"])
        for label, (model, _) in BASELINES.items():
            self.add("download-" + label, lambda label=label: self.download_baseline(label), ("setup",), [self.root / f"baseline-{label}.json"])
            self.add("score-" + label, lambda model=model: self.score(model), ("download-" + label,), [self.results_dir(model) / "summary.md"])
        scores = ["score-" + tag for tag in BENCH_MODELS] + ["score-Qwen3.8-27B", "score-Gemma-4-E4B"]
        self.add("compare-open-models", lambda: self.compare(False), scores, [art / "open_models_compare.md"])
        self.add("compare-jev", lambda: self.compare(True), scores + ["score-Jev"], [art / "jev_compare.md"])
        self.add("latency", lambda: self.python("scripts/bench_latency.py", "--backbone", STUDENT, "--ks", "1", "2", "4", "8", "16", "--repeats", "5", "--device", "cuda"),
                 ("tests",), [art / "bench_latency.json"])
        self.add("async", lambda: self.python("scripts/bench_async.py", "--backbone", STUDENT, "--rates", "5", "20", "50", "--n", "200", "--device", "cuda"),
                 ("tests",), [art / "bench_async.json"])

    def bundle(self, final=False):
        """Logs/results always survive a Python exception. Runtime loss needs download.

        Does not copy HF tokens, environment variables, .git, raw Jev databases,
        cached model weights, or multi-GB extracted feature tensors.
        """
        paths = []
        for pattern in ["*.json", "*.txt", "*.xml", "*.py", "logs/*.log", "coverage/*.json"]:
            paths.extend(self.root.glob(pattern))
        if self.repo.exists():
            for pattern in ["artifacts/**/*.json", "artifacts/*.md", "runs/**/*.jsonl"]:
                paths.extend(self.repo.glob(pattern))
        if self.harness.exists():
            for pattern in ["results/**/*.json", "results/**/*.md", "responses/LICENSE_RESPONSES.*", "LICENSE"]:
                paths.extend(self.harness.glob(pattern))
        paths = sorted({p for p in paths if p.is_file() and p.name != "bundle_manifest.json"})
        manifest = [{"path": str(p.relative_to(self.root)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths]
        write_json(self.root / "bundle_manifest.json", manifest)
        paths = sorted(set(paths + [self.root / "bundle_manifest.json"]))
        target = self.root / "results.zip"
        temporary = target.with_suffix(".tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
            for path in paths:
                z.write(path, path.relative_to(self.root))
        temporary.replace(target)
        if final:
            checkpoints = []
            for name in list(HEADS) + ["lora"]:
                directory = self.repo / "artifacts" / name
                if directory.exists():
                    checkpoints.extend(p for p in directory.iterdir() if p.is_file())
            # Exact source records permit auditing upstream dataset drift.
            records = self.repo / "artifacts" / "feats" / "records.pt"
            if records.exists():
                checkpoints.append(records)
            with zipfile.ZipFile(self.root / "checkpoints.zip", "w", zipfile.ZIP_DEFLATED) as z:
                for path in sorted(checkpoints):
                    z.write(path, path.relative_to(self.root))

    def run(self):
        self.make_plan()
        write_json(self.root / "stage_plan.json", [{"name": n, "dependencies": d, "outputs": [str(p.relative_to(self.root)) for p in o]} for n, _, d, o, _ in self.plan])
        try:
            for entry in self.plan:
                self.stage(*entry)
        finally:
            self.bundle(final=True)
        failed = [n for n, *_ in self.plan if self.status.get(n, {}).get("status") != "succeeded"]
        self.emit("\nRESULTS: " + str(self.root / "results.zip"))
        self.emit("CHECKPOINTS: " + str(self.root / "checkpoints.zip"))
        self.emit("ALL STAGES SUCCEEDED" if not failed else "INCOMPLETE: " + ", ".join(failed))
        return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/content/jex-full-run")
    parser.add_argument("--jev-license-ack", action="store_true", help="Enable Jev response download after license review. Does not waive license restrictions.")
    parser.add_argument("--allow-colab-optional-package-cleanup", action="store_true",
                        help="In a verified hosted Colab runtime only, remove unused TorchAO older than 0.16.0 if it conflicts with pinned PEFT")
    parser.add_argument("--plan-only", action="store_true", help="Print scope only; no cloning, installation, models, or experiments")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(CONFIG, indent=2))
        return 0
    return Runner(args.root, jev_license_ack=args.jev_license_ack,
                  allow_colab_optional_package_cleanup=args.allow_colab_optional_package_cleanup).run()


if __name__ == "__main__":
    raise SystemExit(main())
