import torch
from torch import nn


def accuracy(targets: torch.Tensor, inputs: torch.Tensor) -> float:
    return torch.sum(targets.argmax(-1) == inputs.argmax(-1)) / targets.shape[1]


def mse(target: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    assert (
        target.shape == inputs.shape
    ), f"invalid shape. target is {target.shape}, input is {inputs.shape}. It wants the same shape."
    return torch.mean((target - inputs) ** 2)


def kld(p: torch.Tensor, q: torch.Tensor):
    assert (
        p.shape == q.shape
    ), f"invalid shape. target is {p.shape}, input is {q.shape}. It wants the same shape."
    return p * torch.log2(p / q)


def jsd(p: torch.Tensor, q: torch.Tensor):
    assert (
        p.shape == q.shape
    ), f"invalid shape. target is {p.shape}, input is {q.shape}. It wants the same shape."
    p = torch.clip(p, 0.0001)
    q = torch.clip(q, 0.0001)
    m = 1 / 2 * (p + q)
    return torch.mean(torch.sum(1 / 2 * (kld(p, m) + kld(q, m)), 1))
