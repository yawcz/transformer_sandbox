import torch as t


class Embedding(t.nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device: t.device | None = None, dtype: t.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = t.nn.Parameter(t.empty((num_embeddings, embedding_dim), dtype=dtype, device=device))
        t.nn.init.trunc_normal_(self.weight, a=-3, b=3)

    def forward(self, token_ids: t.Tensor) -> t.Tensor:
        return self.weight[token_ids]
