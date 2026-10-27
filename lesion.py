"""Reversible mean ablation of selected Qwen3-8B MLP neurons."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch

BUNDLE = json.loads(Path(__file__).with_name("lesions.json").read_text())


@contextmanager
def mean_ablate(model, components):
    """Replace [layer, neuron, mean] entries before each MLP down projection.

    Hooks cover every prompt/generated token. Use a fresh generation/cache
    inside each context; removing the hooks restores ordinary computation.
    """
    grouped = {}
    for layer, neuron, mean in components:
        grouped.setdefault(layer, []).append((neuron, mean))
    handles = []
    try:
        for layer, entries in grouped.items():
            module = model.model.layers[layer].mlp.down_proj
            indices = torch.tensor([i for i, _ in entries],
                                   device=module.weight.device)
            means = torch.tensor([v for _, v in entries],
                                 device=module.weight.device,
                                 dtype=module.weight.dtype)

            def replace(module, args, indices=indices, means=means):
                x = args[0].clone()
                x[..., indices] = means.to(dtype=x.dtype)
                return (x, *args[1:])

            handles.append(module.register_forward_pre_hook(replace))
        yield
    finally:
        for handle in handles:
            handle.remove()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", nargs="?", default="Calculate 247 * 38.")
    parser.add_argument("--seed", choices=list(BUNDLE["lesions"]), default="101")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(BUNDLE["model"], revision=BUNDLE["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        BUNDLE["model"], revision=BUNDLE["revision"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(args.device).eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.problem +
          "\nReply with only the exact answer, with no explanation."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    @torch.inference_mode()
    def answer():
        # Each call starts from the original prompt, with no reused KV cache.
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=32,
                             pad_token_id=tokenizer.eos_token_id,
                             use_cache=True, disable_compile=True)
        return tokenizer.decode(ids[0, inputs.input_ids.shape[1]:],
                                skip_special_tokens=True).strip()

    print("healthy: ", answer())
    with mean_ablate(model, BUNDLE["lesions"][args.seed]):
        print("lesioned:", answer())
    print("repaired:", answer())


if __name__ == "__main__":
    main()
