# CogVideoX adapters

Open-EA registers `cogvideox-2b` and `cogvideox-5b` as text-to-video
generators. The adapter downloads `THUDM/CogVideoX-2b` or
`THUDM/CogVideoX-5b` from Hugging Face on first use unless a matching local
directory exists under `checkpoints/`.

Install the model-specific dependencies in the evaluation environment:

```bash
python -m pip install -r eval_agent/eval_models/cogvideox/requirements.txt
```

Run a standalone smoke test from the repository root:

```bash
python eval_agent/eval_models/cogvideox/inference.py \
    --model 2b \
    --prompt "A cat playing with a ball" \
    --output cogvideox_sample.mp4 \
    --seed 42
```

Or select the adapter in the Open-EA pipeline:

```bash
cd eval_agent
python eval_agent_for_vbench_open.py \
    --model cogvideox-2b \
    --user_query "How consistent is the subject throughout the video?"
```

The default generation settings are 49 frames, 8 FPS, 50 denoising steps,
and a fixed seed of 42 in the evaluation adapter. See
`cogvideox_predict.py` for the optional inference parameters.

Model use remains subject to the upstream model-specific terms. Consult the
[CogVideoX-2B model card](https://huggingface.co/THUDM/CogVideoX-2b) and
[CogVideoX-5B model card](https://huggingface.co/THUDM/CogVideoX-5b) before use;
their license terms are not identical.
