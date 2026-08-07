# EA-3B training

This directory contains the cleaned release recipe for training the local
planning backbone used by Open Evaluation Agent (Open-EA): full-parameter
supervised fine-tuning of `Qwen/Qwen2.5-3B-Instruct` on the 10,042-record,
table-context T2V core of EA-CoT-10K.

The published checkpoint is available at
[`open-ea/ea-3b-tab`](https://huggingface.co/open-ea/ea-3b-tab).

## Scope

- The EA-3B training set contains **10,042** history-conditioned T2V records
  covering the VBench tool workflow.
- The separately processed **986-record T2I companion set is not mixed into
  EA-3B training**. It is therefore intentionally absent from the main
  training configuration in this directory.
- The recipe uses full-parameter SFT, not LoRA or QLoRA.
- LLaMA-Factory remains a third-party dependency. Its source is tracked as a
  Git submodule pinned to the reference commit rather than vendored into this
  repository.

This release explicitly registers the optional `system` and `history` fields,
which is the intended history-conditioned setup. The exact historical
`dataset_info.json` snapshot used for the published checkpoint is unavailable,
so this recipe should not be interpreted as a bitwise reconstruction of that
checkpoint.

## Files

| Path | Purpose |
| --- | --- |
| `configs/ea3b_full_sft.yaml` | Canonical EA-3B full-SFT configuration |
| `data/dataset_info.json` | LLaMA-Factory dataset registration and column mapping |
| `deepspeed/zero3.json` | DeepSpeed ZeRO-3 configuration |
| `scripts/train.sh` | Root-relative, four-GPU training launcher |

All paths in the configuration are relative to the Evaluation-Agent
repository root. Run the launcher from the repository root as shown below.

## 1. Initialize and install the pinned training framework

The reference run used LLaMA-Factory commit
[`f00742b078599a7e5295f17c02acfef1b525492d`](https://github.com/hiyouga/LLaMA-Factory/commit/f00742b078599a7e5295f17c02acfef1b525492d)
(`llamafactory==0.9.4.dev0`). Use an isolated training environment; the
top-level `requirements.txt` serves the evaluation stack and is not a lockfile
for model training.

```bash
git submodule update --init --recursive -- LLaMA-Factory

# Install PyTorch appropriate for the local CUDA runtime first.
python -m pip install torch==2.7.1
python -m pip install -e "./LLaMA-Factory[deepspeed]" --no-build-isolation
```

The parent repository records the exact commit above, so `git submodule
update` checks out that revision. Do not switch the submodule to a floating
branch when reproducing the reference setup.

The recorded reference environment used Python 3.10.18, PyTorch 2.7.1,
Transformers 4.52.4, Datasets 3.6.0, Accelerate 1.7.0, DeepSpeed 0.16.9,
PEFT 0.15.2, TRL 0.9.6, and four NVIDIA RTX 6000 Ada Generation GPUs. Exact
package versions are useful for strict reproduction, although compatible
hardware-specific PyTorch builds may be required on other systems.

## 2. Provide the processed training data

The processed EA-CoT-10K dataset is published as
[`open-ea/ea-cot-10k`](https://huggingface.co/datasets/open-ea/ea-cot-10k).
Download its canonical `data/ea_cot_10k.json` artifact to the local path used
by the training registry:

```bash
curl --location --fail \
  https://huggingface.co/datasets/open-ea/ea-cot-10k/resolve/main/data/ea_cot_10k.json \
  --output training/data/evaluation_agent_cot_dataset_table.json
```

The separately published
[`open-ea/ea-cot-t2i`](https://huggingface.co/datasets/open-ea/ea-cot-t2i)
companion is not part of this training recipe. The expected local T2V path is:

```text
training/data/evaluation_agent_cot_dataset_table.json
```

Each record must use the following Alpaca-style fields:

```json
{
  "instruction": "current user request or observation",
  "input": "",
  "output": "next reasoning/tool decision or final summary",
  "system": "evaluation protocol",
  "history": [["previous user turn", "previous assistant turn"]]
}
```

`data/dataset_info.json` explicitly maps all five columns. This is important:
LLaMA-Factory infers `instruction`, `input`, and `output`, but optional
`system` and `history` fields are only consumed when they are registered.

The dataset is prefix-expanded: every row targets the next assistant turn while
retaining earlier turns as context. Accordingly, the release configuration sets
`mask_history: true`. LLaMA-Factory still includes the full history in the model
input, but computes supervised loss only on the current response. This avoids
repeatedly weighting earlier assistant turns and prioritizes the current target
when a long conversation reaches the sequence cutoff.

The dataset file is intentionally ignored by Git. Before starting a long run,
the following optional check verifies the record count and schema:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("training/data/evaluation_agent_cot_dataset_table.json")
records = json.loads(path.read_text(encoding="utf-8"))
required = {"instruction", "input", "output", "system", "history"}
assert len(records) == 10_042, len(records)
assert all(required <= record.keys() for record in records)
print(f"validated {len(records):,} records")
PY
```

## 3. Train EA-3B

From the Evaluation-Agent repository root, run:

```bash
bash training/scripts/train.sh
```

The canonical run uses four GPUs. The launcher defaults to devices `0,1,2,3`
and allows the standard environment variable to override them:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash training/scripts/train.sh
```

Using a different number of GPUs changes the effective batch size unless
`gradient_accumulation_steps` is adjusted accordingly.

## Canonical hyperparameters

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Training stage | supervised fine-tuning (`sft`) |
| Fine-tuning type | full parameter |
| Dataset | `ea_cot_10k_table` (10,042 T2V records) |
| Template | `qwen` |
| Sequence cutoff | 4,096 tokens |
| History loss | Current assistant turn only (`mask_history: true`) |
| Distributed strategy | DeepSpeed ZeRO-3 |
| Precision | bfloat16 |
| Per-device batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 64 on four GPUs |
| Epochs | 3 |
| Optimizer | AdamW (PyTorch) |
| Learning rate | `1.0e-5` |
| Scheduler | cosine |
| Warmup ratio | 0.1 |
| Gradient checkpointing | enabled |
| Seed | 42 |
| Checkpoint interval | 200 steps |

The configuration does not create a validation split or run evaluation during
training, matching the reference run. Training outputs are written under
`training/outputs/ea3b_full_sft/` and ignored by Git. The output directory is
configured to be overwritten, so move or rename an existing run before
launching another one if it must be preserved.

Experiment tracking is disabled in the release configuration to avoid making
an external account a runtime requirement. To reproduce the original W&B
logging setup, install `wandb==0.20.1`, authenticate with W&B, and change
`report_to` in `configs/ea3b_full_sft.yaml` from `none` to `wandb`.
