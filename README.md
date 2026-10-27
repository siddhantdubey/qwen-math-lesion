# Qwen math lesion

Weaken arithmetic in [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) by freezing
16–32 MLP neuron activations. Reversible hooks; weights stay unchanged.

```sh
pip install -r requirements.txt
python lesion.py "Calculate 247 * 38."
python lesion.py --seed 211 "Calculate 247 * 38."
```

Prints healthy → lesioned → repaired. Default: 16 neurons, seed 101.
Python 3.10–3.13; ~16 GB for weights plus runtime memory. `lesions.json` holds
all five original presets, replacement means, artifact hashes and pinned model
revision. Each generation starts with a fresh cache.

**[Attribution patching](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching):**
estimate which neurons help correct math answers using
`(activation − clean mean) × gradient`. The gradient says how sensitive the
correct answer's log probability is to that activation. Subtract average
non-math scores, rank neurons, then test actual ablations.

**Mean ablation:** replace a selected neuron's changing value with its usual
clean-data average, before `mlp.down_proj`. It stops carrying input-specific
information. Removing the hooks restores normal computation.

Original development accuracy with the default lesion: arithmetic **91.3% →
60.7%**, factual **89.8% → 88.2%**, language **99.3% → 99.5%**. This weakens
arithmetic; it does not erase all math or prove these neurons only do math.
Extracted from the 2026-10-25 lesion experiment.
