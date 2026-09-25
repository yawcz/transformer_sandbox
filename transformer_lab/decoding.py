import torch as t

from transformer_lab.softmax import softmax
from transformer_lab.tokenizer import Tokenizer
from transformer_lab.transformer_lm import TransformerLM


def decoder(
    model: TransformerLM, tokenizer: Tokenizer, prompt: str, temperature: float, max_tokens: int, top_p: float = 1
) -> str:
    tokenized_prompt = tokenizer.encode(prompt)
    if not tokenized_prompt:
        raise ValueError("prompt must contain at least one token")

    model.eval()

    with t.inference_mode():
        while len(tokenized_prompt) < max_tokens:
            context = tokenized_prompt[-model.context_length :]
            probs = softmax(model(context)[-1] / temperature, 0).tolist()
            if top_p < 1:
                sorted_zipped_probs = sorted(zip(probs, range(len(probs))), reverse=True)
                new_probs: list[tuple[float, int]] = []
                cum_p = 0
                for prob, token in sorted_zipped_probs:
                    cum_p += prob
                    new_probs.append((prob, token))
                    if cum_p >= top_p:
                        break
                dist = t.distributions.Categorical(t.tensor([x[0] for x in new_probs]))
                tokenized_prompt.append(sorted_zipped_probs[int(dist.sample().item())][1])
            else:
                dist = t.distributions.Categorical(t.tensor(probs))
                tokenized_prompt.append(int(dist.sample().item()))

            if tokenized_prompt[-1] == 0:
                break

    model.train()

    return tokenizer.decode(tokenized_prompt)
