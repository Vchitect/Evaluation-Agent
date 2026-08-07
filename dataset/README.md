# Datasets

This repository documents two distinct data resources: the original 100-query dataset used to express open-ended evaluation needs, and the EA-CoT training corpus introduced by Open-EA.

## Open-Ended User Query Dataset

We compiled the final set of 100 open-ended user queries into [`open_ended_user_questions.json`](./open_ended_user_questions.json). The accompanying [`open_ended_user_questions_summary.json`](./open_ended_user_questions_summary.json) contains the dataset summary metadata.

This query collection is an evaluation input resource. It is separate from the history-conditioned supervision used to train EA-3B.

## EA-CoT Training Corpus (Open-EA)

EA-CoT is constructed from recorded, multi-round Evaluation Agent rollouts. Each rollout is unfolded into step-level next-turn records that teach a planner either to make the next reasoning-and-tool decision or to synthesize the final evidence-grounded summary.

### Composition

| Split / artifact | Modality | Records | Tool-decision records | Summary records | Tool coverage | Used for current EA-3B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [EA-CoT-10K T2V core](https://huggingface.co/datasets/open-ea/ea-cot-10k) (table context) | Text-to-video | 10,042 | 7,494 | 2,548 | 15 VBench tools | Yes |
| [EA-CoT-T2I companion](https://huggingface.co/datasets/open-ea/ea-cot-t2i) | Text-to-image | 986 | 739 | 247 | 4 T2I-CompBench tools | No |

The T2I companion is processed and published separately; it is not mixed into the current EA-3B training run.

### Record Schema

Every processed record follows the same Alpaca-style schema:

| Field | Description |
| --- | --- |
| `instruction` | The initial evaluation request or the latest `<information>...</information>` observation. |
| `input` | The Alpaca input field; empty in the current processed artifacts. |
| `output` | Either the next `<think>...</think><tool>...</tool>` decision or the terminal `<think>...</think><summary>...</summary>` response. |
| `system` | The Evaluation Agent protocol, available tools, and structured-output rules. |
| `history` | Earlier instruction/response pairs retained as context for the current step. |

Each row is a **history-conditioned step**, not an independent full trajectory. The processed SFT artifacts are textual records containing prompts, observations, tool decisions, and summaries; they are not a bundle of the source images or videos.

### Access and Reproduction

- **Official Hugging Face organization:** [`open-ea`](https://huggingface.co/open-ea)
- **EA-CoT-10K T2V core:** [`open-ea/ea-cot-10k`](https://huggingface.co/datasets/open-ea/ea-cot-10k), file `data/ea_cot_10k.json`
- **EA-CoT-T2I companion:** [`open-ea/ea-cot-t2i`](https://huggingface.co/datasets/open-ea/ea-cot-t2i), file `data/ea_cot_t2i.json`
- **Data construction code:** [`../data_construction/`](../data_construction/)
- **Training configuration:** [`../training/`](../training/)
- **EA-3B checkpoint:** [`open-ea/ea-3b-tab`](https://huggingface.co/open-ea/ea-3b-tab)