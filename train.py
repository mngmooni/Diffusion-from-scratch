"""Train the U-Net on Fashion-MNIST (from random initialization)."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from diffusion import DiffusionUNet
from diffusion.sampling import DiffusionSchedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("runs/model_5000.pt"))
    args = parser.parse_args()

    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    data = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(42)).tolist()
    train_loader = DataLoader(Subset(data, indices[:55000]), batch_size=64, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(Subset(data, indices[55000:]), batch_size=64, shuffle=False,
                            num_workers=2, pin_memory=(device.type == "cuda"))

    model = DiffusionUNet().to(device)
    diffusion = DiffusionSchedule(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    iterator = iter(train_loader)
    losses = []

    for step in range(1, args.steps + 1):
        try:
            clean, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            clean, _ = next(iterator)
        clean = clean.to(device)
        t = torch.randint(0, diffusion.steps, (clean.shape[0],), device=device)
        noise = torch.randn_like(clean)
        noisy = diffusion.q_sample(clean, t, noise)
        loss = F.mse_loss(model(noisy, t), noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % 500 == 0 or step == args.steps:
            print(f"step {step:5d}  noise MSE {np.mean(losses[-500:]):.4f}")

    model.eval()
    generator = torch.Generator(device=device).manual_seed(31415)
    sq_error, elements = 0.0, 0
    with torch.no_grad():
        for batch_no, (clean, _) in enumerate(val_loader):
            if batch_no == 8:
                break
            clean = clean.to(device)
            t = torch.randint(0, diffusion.steps, (len(clean),), device=device, generator=generator)
            noise = torch.randn(clean.shape, device=device, generator=generator)
            sq_error += F.mse_loss(model(diffusion.q_sample(clean, t, noise), t), noise, reduction="sum").item()
            elements += noise.numel()
    print(f"validation noise MSE: {sq_error / elements:.5f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"Won't overwrite {args.out}")
    torch.save({"model": model.state_dict(), "step": args.steps,
                "betas": diffusion.betas.cpu(), "split_seed": 42}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
