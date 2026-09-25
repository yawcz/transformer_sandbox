import math
from collections.abc import Callable

import torch as t


class AdamW(t.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-3,
        eps: float = 1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    @t.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = t.zeros_like(p)
                    state["v"] = t.zeros_like(p)

                t_opt = state["t"]
                m = state["m"]
                v = state["v"]

                grad = p.grad

                lr_t = lr * math.sqrt(1 - betas[1] ** t_opt) / (1 - betas[0] ** t_opt)

                p.mul_(1 - lr * weight_decay)

                m.lerp_(grad, 1 - betas[0])
                v.mul_(betas[1])
                v.addcmul_(grad, grad, value=1 - betas[1])

                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)

                state["t"] = t_opt + 1

        return loss
