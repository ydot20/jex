# Complete configured Colab run

Use [`notebooks/jex_gpu_full.ipynb`](../notebooks/jex_gpu_full.ipynb) for the original GPU notebook's complete experiment configuration, with checked subprocesses, resumable stage status, and downloadable evidence. The original [`jex_gpu.ipynb`](../notebooks/jex_gpu.ipynb) is unchanged.

## Run

1. Upload `notebooks/jex_gpu_full.ipynb` to Colab and select a GPU runtime.
2. Review the [released response-data license](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking/blob/6bbdeb33474849b6de2f0cccc9f5e19756abd67e/responses/LICENSE_RESPONSES.md). `JEV_RESPONSE_LICENSE_ACK` defaults to `False`. Enable it only for a compliant intended use; it does not waive restrictions. With it disabled, Jev download/scoring/comparison are reported as blocked and independent work continues.
3. Choose **Run all**. Keep the full output and download `results.zip` and `checkpoints.zip` before the runtime disappears.

No GitHub token, Drive mount, or paid model API is required. Gated datasets may remain unavailable and must be disclosed in the final coverage report. This runner does not accept dataset access terms or request credentials automatically.

The same runner can be invoked in a prepared GPU environment:

```bash
python scripts/colab_full_run.py --root /content/jex-full-run
# Only after reviewing the license and intended use:
python scripts/colab_full_run.py --root /content/jex-full-run --jev-license-ack
```

To inspect the fixed configuration without installing anything or running experiments:

```bash
python scripts/colab_full_run.py --plan-only
```

## What “complete” means

The runner executes all stages at the original notebook defaults:

- Student: Qwen2.5-1.5B-Instruct; teacher: Qwen2.5-7B-Instruct, 4-bit.
- 13 training tasks and four held-out tasks. Train/dev/in-domain eval/held-out eval: **400/50/150/400 records per task**, totaling 9,400 records if every source is available.
- Five heads: primary, gold-only, no-RLCD, no-memory, no-memory with prior-KL 1.0.
- LoRA: **3,000 training records**, one epoch, original batch-four and dtype defaults.
- Four benchmark models: Qwen2.5-1.5B zero-shot, trained LoRA, Qwen3.5-4B zero-shot, Qwen3.5-9B zero-shot in 4-bit.
- **30 examples per benchmark dataset**, with the same limit for released-response baselines and local models.
- Latency: K=1/2/4/8/16, `--repeats 5`. The original latency script still uses one measured repetition for its autoregressive and K-separate baselines.
- Async: arrival rates 5/20/50, 200 arrivals per rate/server combination.

This is the **full configured notebook run**, not evaluation on every example of every benchmark dataset. A full-dataset extension is intentionally separate. It needs a separately specified generation/scoring protocol, consistent limits across models, adequate resources, and a fresh output directory. Setting `--limit 0` does not request a full evaluation. No full-dataset extension is started by this notebook.

## Failure, recovery, and coverage

Each subprocess retains stdout/stderr in its stage log and fails on a nonzero exit code. Failed dependencies block downstream stages; unrelated experiments continue. The runner exits nonzero if any declared stage failed or remains blocked. A successful subprocess does **not** prove complete dataset coverage: the upstream benchmark code can skip unavailable configurations or schema-rejected requests.

Rerun in the same runtime and root directory to retry unsuccessful stages. Completed stages with nonempty expected outputs are retained, and extraction reuses the exact saved source records. A changed configuration is refused in an existing root; use a new root for a different experiment. Resume is same-runtime recovery, not a guarantee against deleted or modified artifacts. Training itself is not checkpointed mid-stage.

The upstream harness contains published full-sample result JSON files. On first setup, those are preserved in `published-results-reference-only/`, outside the scoring directory. Fresh evaluations cannot silently inherit their scores. Per-model coverage files retain answered/example counts and skipped configurations; comparison metadata lists the common-task intersection and excluded tasks. Inspect these files and logs before reporting a mean or saying all 37 tasks completed.

## Outputs

The run root contains:

- `run_config.json`, `stage_plan.json`, `stage_status.json`: exact settings, dependency plan, outcomes, timings, and errors.
- `logs/`: complete per-stage command output.
- `pip-freeze.txt`, `environment.txt`, `nvidia-smi.txt`: resolved packages and hardware.
- `coverage/` and comparison coverage JSON: scored-task coverage and intersection details.
- `results.zip`: logs, metadata, result tables/JSON, local-model response JSONL, licenses, attribution, and a SHA-256 manifest. Refreshed after each stage.
- `checkpoints.zip`: available trained head/adapter files and exact source records, assembled when the runner exits normally or handles an exception.

Archives omit source response SQLite databases, credentials, environment-variable dumps, `.git`, cached backbone weights, and large extracted feature tensors. Hard runtime termination can prevent final checkpoint packaging. Colab storage is temporary; download results promptly. Do not republish source records without reviewing their dataset licenses.

## Reproducibility and resource caveats

- jex is pinned to `c494049aa3af480abc5ec502a10a783b26c2a71b`.
- The harness is pinned to `6bbdeb33474849b6de2f0cccc9f5e19756abd67e`.
- Transformers 5.14.1 matches the harness's exact open-model requirement. PEFT 0.21.2 supplies the missing LoRA dependency; flash-linear-attention 0.5.2 is installed after the CPU tiny-model tests. Colab's CUDA PyTorch installation is retained subject to dependency requirements; resolved versions are recorded.
- Python, NumPy and torch seeds are set. CUDA nondeterminism and unpinned upstream model/dataset Hub revisions still prevent a claim of bitwise reproducibility.
- Original T4 LoRA uses float32 with batch four and no gradient checkpointing, so it may exhaust GPU memory. The runner never silently reduces model sizes, sample counts, batches, epochs or benchmark limits.
- The existing benchmark truncates state to 3,072 tokens; released baseline responses may have used longer state. Model comparisons must disclose this methodological difference.
- Compatibility, resource requirements, scores, and execution time must be established by the actual Colab run. Orchestration tests are not GPU experiment results.

## Response-data use and attribution

Released Jev responses are used only for evaluation, after training. They are never supplied as training labels, soft targets, rewards, or optimization input. Their license permits research/evaluation/benchmarking and prohibits distillation and developing or facilitating a similar or competing product. Qwen3.8 and Gemma responses use Apache 2.0. Output archives retain the unchanged response-data license; source databases are not redistributed.

Reference: Tobias Deußer, Lorenz Sparrenberg and Rafet Sifa, *Evaluating and Benchmarking the System One Model Jev*, arXiv:2609.37647. Response dataset: [DOI 10.5281/zenodo.23039006](https://doi.org/10.5281/zenodo.23039006).

## Develop the runner

The notebook embeds the runner and verifies its hash, so uploads do not depend on an uncommitted remote branch. Regenerate it after changing the script:

```bash
python scripts/build_colab_notebook.py
python -m unittest discover -s tests -p test_colab_runner.py -v
```

These standard-library tests do not import torch, fetch datasets, or run model experiments. The repository's normal pytest suite still runs inside the GPU notebook after dependencies are installed and before flash-linear-attention is installed.
