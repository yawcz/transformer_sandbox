import torch as t

from transformer_lab.hookpoint import HookPoint
from transformer_lab.tokenizer import Tokenizer
from transformer_lab.transformer_lm import TransformerLM


def get_hook(module_name, cache, callback=None):
    def extract_activations(module, input, output):
        if callback is not None:
            cache[module_name] = callback(module, input, output)
            return cache[module_name]
        else:
            cache[module_name] = output

    return extract_activations


def run_with_cache(
    model: TransformerLM,
    tokenizer: Tokenizer,
    x: str | t.Tensor,
    device: t.device,
    callbacks={},
) -> tuple[t.Tensor, dict]:
    if isinstance(x, str):
        # model is intentionally run without a batch dimension
        x = t.tensor(tokenizer.encode(x), dtype=t.long, device=device)
    else:
        # x is a tensor; batch dimension may or may not exist
        x = x.to(device)

    model.eval()

    cache = {}
    handles = []

    try:
        for name, module in model.named_modules():
            if isinstance(module, HookPoint):
                handles.append(module.register_forward_hook(get_hook(name, cache, callbacks.get(name, None))))

        with t.inference_mode():
            output = model(x)
    finally:
        for handle in handles:
            handle.remove()

    return output, cache
