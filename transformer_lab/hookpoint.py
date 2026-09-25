import torch as t


class HookPoint(t.nn.Module):
    def forward(self, x):
        return x
