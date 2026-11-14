"""DDPM and DDIM samplers using the same 200-step linear beta schedule."""

import torch


class DiffusionSchedule:
    def __init__(self, device, steps=200):
        self.steps = steps
        self.betas = torch.linspace(1e-4, 0.05, steps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        previous = torch.cat([torch.ones_like(self.alpha_bars[:1]), self.alpha_bars[:-1]])
        self.posterior_variance = self.betas * (1 - previous) / (1 - self.alpha_bars)

    def q_sample(self, clean, t, noise):
        a = self.alpha_bars[t].view(-1, 1, 1, 1)
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    def ddim_schedule(self, nfe):
        if not isinstance(nfe, int) or not 1 <= nfe <= self.steps:
            raise ValueError(f"NFE must be between 1 and {self.steps}")
        times = torch.linspace(self.steps - 1, 0, nfe).round().long().tolist()
        times = sorted(set(times), reverse=True)
        return list(zip(times, times[1:] + [-1]))


@torch.no_grad()
def ddpm_sample(model, initial, schedule):
    x = initial.clone()
    for i in reversed(range(schedule.steps)):
        t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
        eps = model(x, t)
        mean = (x - schedule.betas[i] * eps / (1 - schedule.alpha_bars[i]).sqrt()) / schedule.alphas[i].sqrt()
        x = mean + schedule.posterior_variance[i].sqrt() * torch.randn_like(x) if i > 0 else mean
    return x


@torch.no_grad()
def ddim_sample(model, initial, schedule, nfe=50):
    x = initial.clone()
    for i, j in schedule.ddim_schedule(nfe):
        t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
        eps = model(x, t)
        a_t = schedule.alpha_bars[i]
        a_next = schedule.alpha_bars[j] if j >= 0 else x.new_tensor(1.0)
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
    return x
