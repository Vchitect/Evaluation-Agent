# EA-CoT data construction

This directory contains the release-facing converter for building EA-CoT
instruction-tuning records from **already curated** Evaluation Agent trajectory
JSON files.

The documented processed artifacts have two separately processed parts:

| Split | Modality | Step-level records | Tool-decision records | Summary records | Used to train EA-3B |
| --- | --- | ---: | ---: | ---: | --- |
| [EA-CoT-10K core](https://huggingface.co/datasets/open-ea/ea-cot-10k) | Text-to-video (T2V) | 10,042 | 7,494 | 2,548 | Yes |
| [EA-CoT-T2I companion](https://huggingface.co/datasets/open-ea/ea-cot-t2i) | Text-to-image (T2I) | 986 | 739 | 247 | No |

The 986-record T2I companion set follows the same record structure but was not
mixed into the EA-3B training run.

These counts describe the documented processed artifacts; the converter does
not force a target count. Its output is determined by the supplied curated
manifest and the explicit deduplication option.

## Scope and curation boundary

`build_ea_cot.py` performs deterministic **format conversion**. It does not
sample images or videos, run VBench/T2I-CompBench metrics, judge visual
faithfulness, or decide which trajectories belong in the release.

The input directory must therefore contain the release-curated trajectories.
The original release process included curation outside this converter, but this
repository does not contain the original two-stage filtering implementation.
Do not describe this script as reproducing score-based or MLLM-judge filtering.

## Requirements

- Python 3.9 or newer.
- No third-party Python packages are required by this converter.
- The repository must provide these prompt entries through
  `eval_agent/system_prompts.py`:
  - `eval-agent-vbench-training-sys_v1`
  - `eval-agent-t2i-training-sys`
  - `eval-agent-format-sys`
- Score reference tables are read from:
  - `eval_agent/vbench_dimension_scores.tsv` for T2V;
  - `eval_agent/t2i_comp_dimension_scores.tsv` for T2I.

Relative CLI paths are resolved from the repository root, so commands behave
the same regardless of the caller's current working directory.

## Input trajectory format

Each source JSON file is a list containing a user query, one or more tool steps,
and one final summary step:

```json
[
  "How accurately does the model generate the requested scene?",
  {
    "Sub-aspect": "Simple scene composition",
    "Tool": "Scene",
    "Thought": "I will establish a simple baseline.",
    "eval_results": {
      "score": [
        0.75,
        [
          {
            "prompt": "A quiet park in autumn",
            "video_results": 0.75
          }
        ]
      ]
    }
  },
  {
    "Thought": "The available evidence is sufficient.",
    "Analysis": "The model follows the requested scene reliably.",
    "Summary": "Overall performance is strong on this probe."
  },
  [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
]
```

For T2I trajectories, per-prompt scores use `image_results` instead of
`video_results`. Lower-case variants such as `tool`, `thought`, `sub_aspect`,
`analysis`, and `summary` are also accepted.

`Analysis` and `Summary` should normally be strings. For compatibility with
legacy trajectories, non-empty JSON objects or lists in these two fields are
serialized using the same Python string representation as the original release
builder.

The final list is removed **only** when it is an OpenAI-style chat messages list
whose entries have `role` and `content`. No arbitrary last item is discarded.
Unexpected items cause that source file to be reported as an error.

## Build commands

T2V core:

```bash
python data_construction/build_ea_cot.py \
  --modality t2v \
  --input-dir /path/to/curated/t2v-trajectories \
  --output training/data/evaluation_agent_cot_dataset_table.json \
  --score-table eval_agent/vbench_dimension_scores.tsv \
  --seed 0 \
  --report training/data/evaluation_agent_cot_dataset_table.report.json
```

T2I companion:

```bash
python data_construction/build_ea_cot.py \
  --modality t2i \
  --input-dir /path/to/curated/t2i-trajectories \
  --output training/data/evaluation_agent_cot_dataset_t2i.json \
  --score-table eval_agent/t2i_comp_dimension_scores.tsv \
  --seed 0 \
  --report training/data/evaluation_agent_cot_dataset_t2i.report.json
```

The T2V output path above is the location consumed by the canonical training
recipe. The T2I file is kept beside it for convenience but is not registered in
`training/data/dataset_info.json` and is not used to train EA-3B.

`--score-table` can be omitted to use the modality-specific default shown
above. Source files are discovered recursively and processed in stable path
order. `--seed` makes the legacy sub-aspect wording variation reproducible.

The converter continues past malformed source files, prints each error to
stderr, and summarizes processed files, skipped duplicates, output records, and
errors. It still writes the successfully converted records and `--report`
output for diagnosis, but exits nonzero if any source failed. Always inspect the
report and verify the expected record counts before release.

## Optional exact-source deduplication

Exact source-content deduplication is deliberately disabled by default. This
avoids silently changing compatibility with a reported reference artifact such
as the 10,042-record T2V core.

To skip byte-for-byte duplicate source JSON files, opt in explicitly:

```bash
python data_construction/build_ea_cot.py \
  --modality t2v \
  --input-dir /path/to/curated/t2v-trajectories \
  --output training/data/evaluation_agent_cot_dataset_deduplicated.json \
  --deduplicate-source-content
```

The first file in stable path order is retained for each SHA-256 source digest.
This option changes dataset membership and may change the final record count;
record the choice in the dataset version and release manifest.

## Output format

Each trajectory is unfolded into one history-conditioned record per decision
step. Every output row has the LLaMA-Factory Alpaca fields:

```json
{
  "instruction": "the user query or previous <information> observation",
  "input": "",
  "output": "<think>...</think><tool>...</tool>",
  "system": "the modality training protocol plus tag grammar",
  "history": [["earlier instruction", "earlier assistant output"]]
}
```

Intermediate targets use `<think>` and `<tool>`. Tool scores and their scoring
reference are returned as the next `<information>` instruction. The terminal
target uses `<think>` and `<summary>`. Media paths and media files are not copied
into the Alpaca output.
