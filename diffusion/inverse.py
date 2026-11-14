"""Measurement operators and normalized-gradient guided DDIM."""

import torch
from torch import nn
from torch.nn import functional as F
from .sampling import ddim_sample


def center_mask(x):
    mask = torch.ones_like(x)
    mask[:, :, 10:22, 10:22] = 0
    return mask


def project_observed(x, y, mask):
    return (mask * y + (1 - mask) * x).clamp(-1, 1)


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=9, sigma=1.5):
        super().__init__()
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gaussian = torch.exp(-coords.square() / (2 * sigma * sigma))
        gaussian /= gaussian.sum()
        self.register_buffer("kernel", (gaussian[:, None] * gaussian[None, :]).view(1, 1, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x):
        c = x.shape[1]
        kernel = self.kernel.to(dtype=x.dtype).expand(c, 1, -1, -1)
        return F.conv2d(F.pad(x, (self.padding,) * 4, mode="reflect"), kernel, groups=c)


@torch.no_grad()
def inpaint_final(model, initial, schedule, y, mask, nfe):
    return project_observed(ddim_sample(model, initial, schedule, nfe), y, mask)


@torch.no_grad()
def inpaint_iterative(model, initial, schedule, y, mask, nfe):
    x = initial.clone()
    for i, j in schedule.ddim_schedule(nfe):
        t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
        eps = model(x, t)
        a_t = schedule.alpha_bars[i]
        a_next = schedule.alpha_bars[j] if j >= 0 else x.new_tensor(1.0)
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        corrected = mask * y + (1 - mask) * x0
        x = a_next.sqrt() * corrected + (1 - a_next).sqrt() * eps
    return project_observed(x, y, mask)


def guided_ddim(model, initial, schedule, operator, y, nfe, strength):
    """Heuristic guided DDIM, not an exact implementation of DPS.

    The gradient of the per-image L2 measurement residual is normalized
    before every deterministic DDIM step. No hard projection is applied here.
    """
    if strength < 0:
        raise ValueError("strength must be nonnegative")
    x = initial.clone()
    for i, j in schedule.ddim_schedule(nfe):
        t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
        a_t = schedule.alpha_bars[i]
        a_next = schedule.alpha_bars[j] if j >= 0 else x.new_tensor(1.0)
        with torch.enable_grad():
            x_input = x.detach().requires_grad_(True)
            eps = model(x_input, t)
            x0 = (x_input - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            residual = operator(x0) - y
            loss = torch.linalg.vector_norm(residual.flatten(1), dim=1).sum()
            gradient = torch.autograd.grad(loss, x_input)[0]
        with torch.no_grad():
            norm = gradient.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1, 1)
            next_x = a_next.sqrt() * x0.detach() + (1 - a_next).sqrt() * eps.detach()
            x = (next_x - strength * gradient / norm).detach()
    return x.clamp(-1, 1)
