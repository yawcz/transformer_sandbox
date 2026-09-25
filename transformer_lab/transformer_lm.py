import torch as t
from jaxtyping import Float, Int

from transformer_lab.embedding import Embedding
from transformer_lab.linear import Linear
from transformer_lab.rmsnorm import RMSNorm
from transformer_lab.transformer_block import TransformerBlock


class TransformerLM(t.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: t.device | None = None,
        dtype: t.dtype | None = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = t.nn.Sequential(
            *[
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self, in_features: Int[t.Tensor, " batch_size sequence_length"]
    ) -> Float[t.Tensor, " batch_size sequence_length vocab_size"]:
        embedded_features = self.token_embeddings(in_features)
        after_transformer_layers = self.layers(embedded_features)
        return self.lm_head(self.ln_final(after_transformer_layers))
