# Show-1 adapter

Open-EA registers `show1` as a text-to-video generator. The small adapter is
included here. The upstream Show-1 source is tracked as a Git submodule pinned
to commit
[`89148aff866a87d439fe7cf428ea37cebdb6b997`](https://github.com/showlab/Show-1/commit/89148aff866a87d439fe7cf428ea37cebdb6b997);
model checkpoints are not included.

From the Evaluation-Agent repository root, initialize the pinned implementation
and install its dependencies:

```bash
git submodule update --init --recursive -- eval_agent/eval_models/show1/Show-1
python -m pip install -r eval_agent/eval_models/show1/Show-1/requirements.txt
python -m pip install -r eval_agent/eval_models/show1/requirements.txt
```

Show-1 pins an older Diffusers release, so use an isolated model environment.
The adapter requirements cap `huggingface-hub` below 0.26 because newer Hub
clients removed an API required by that Diffusers version.

The model weights are downloaded from their upstream Hugging Face repositories
on first use. Some checkpoints, including DeepFloyd IF components, may require
accepting their model license before download.

After setup, select the adapter in the Open-EA pipeline:

```bash
cd eval_agent
python eval_agent_for_vbench_open.py \
    --model show1 \
    --user_query "How smooth and temporally consistent is the generated motion?"
```

The adapter runs the base, interpolation, and two super-resolution stages by
default. These stages use CPU offloading but still require substantial GPU
memory and inference time. Show-1 is distributed upstream under
[CC BY-NC 4.0](https://github.com/showlab/Show-1/blob/89148aff866a87d439fe7cf428ea37cebdb6b997/LICENSE), which
restricts use to non-commercial purposes; the individual model dependencies
may impose additional terms.
