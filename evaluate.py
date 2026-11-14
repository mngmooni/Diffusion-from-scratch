"""Re-run the fixed 256-image, 3-seed evaluation using a saved checkpoint."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from diffusion import DiffusionUNet
from diffusion.sampling import DiffusionSchedule, ddim_sample
from diffusion.inverse import GaussianBlur, center_mask, project_observed, inpaint_final, inpaint_iterative, guided_ddim


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model_5000.pt"))
    parser.add_argument("--out", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--images", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DiffusionUNet().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    schedule = DiffusionSchedule(device)
    assert torch.allclose(checkpoint["betas"], schedule.betas.cpu())
    blur = GaussianBlur().to(device)

    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x * 2 - 1)])
    test = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
    indices = torch.randperm(len(test), generator=torch.Generator().manual_seed(2027))[:args.images].tolist()
    loader = DataLoader(Subset(test, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=(device.type == "cuda"))
    rows, timings = [], []

    def timed_call(fn):
        sync(device)
        started = time.perf_counter()
        image = fn()
        sync(device)
        return image.detach().clamp(-1, 1), time.perf_counter() - started

    for seed in [9000, 9001, 9002]:
        for nfe in [20, 50, 100]:
            elapsed = {}
            for batch_idx, (original, _) in enumerate(loader):
                original = original.to(device)
                ids = indices[batch_idx * args.batch_size:batch_idx * args.batch_size + len(original)]
                generator = torch.Generator(device=device).manual_seed(seed + batch_idx)
                noise = torch.randn(original.shape, device=device, generator=generator)
                mask = center_mask(original)
                observed = mask * original
                with torch.no_grad():
                    blurred = blur(original)

                methods = [
                    ("inpainting", "Final Projection", 0, lambda: inpaint_final(model, noise, schedule, observed, mask, nfe)),
                    ("inpainting", "Iterative Projection", 0, lambda: inpaint_iterative(model, noise, schedule, observed, mask, nfe)),
                ]
                for budget in [10, 25, 40]:
                    methods.append(("inpainting", f"Gradient B={budget}", budget,
                                    lambda b=budget: project_observed(guided_ddim(model, noise, schedule,
                                                           lambda x: mask * x, observed, nfe, b / nfe), observed, mask)))
                methods.extend([
                    ("deblurring", "Blurred Observation", 0, lambda: blurred),
                    ("deblurring", "Unconditional DDIM", 0, lambda: ddim_sample(model, noise, schedule, nfe)),
                ])
                for budget in [25, 50]:
                    methods.append(("deblurring", f"Gradient B={budget}", budget,
                                    lambda b=budget: guided_ddim(model, noise, schedule, blur, blurred, nfe, b / nfe)))

                for task, method, budget, fn in methods:
                    reconstruction, seconds = timed_call(fn)
                    elapsed[(task, method)] = elapsed.get((task, method), 0) + seconds
                    errors = (reconstruction - original).square().flatten(1).mean(1)
                    if task == "inpainting":
                        missing = (1 - mask)
                        missing_mse = ((reconstruction - original).square() * missing).flatten(1).sum(1) / missing.flatten(1).sum(1)
                        observed_mae = ((reconstruction - original).abs() * mask).flatten(1).sum(1) / mask.flatten(1).sum(1)
                        truth_np = original[:, 0].detach().cpu().numpy()
                        recon_np = reconstruction[:, 0].detach().cpu().numpy()
                        ssim = [structural_similarity(a, b, data_range=2.0) for a, b in zip(truth_np, recon_np)]
                        for j, idx in enumerate(ids):
                            mse = missing_mse[j].item()
                            rows.append(dict(task=task, test_index=idx, seed=seed, nfe=nfe, method=method,
                                             budget=float(budget), **{"lambda": float(budget / nfe) if budget else 0.0},
                                             missing_mse=mse, missing_psnr=10 * np.log10(4.0 / max(mse, 1e-12)),
                                             full_ssim=ssim[j], observed_mae=observed_mae[j].item()))
                    else:
                        with torch.no_grad():
                            measurement_mse = (blur(reconstruction) - blurred).square().flatten(1).mean(1)
                        for j, idx in enumerate(ids):
                            rows.append(dict(task=task, test_index=idx, seed=seed, nfe=nfe, method=method,
                                             budget=float(budget), **{"lambda": float(budget / nfe) if budget else 0.0},
                                             reconstruction_mse=errors[j].item(), measurement_mse=measurement_mse[j].item()))

            for (task, method), seconds in elapsed.items():
                timings.append(dict(task=task, method=method, seed=seed, nfe=nfe, elapsed_sec=seconds))
            print(f"finished seed={seed}, NFE={nfe}")

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "per_image.csv", index=False)
    pd.DataFrame(timings).to_csv(args.out / "runtime.csv", index=False)
    df.groupby(["task", "nfe", "method"], as_index=False).mean(numeric_only=True).to_csv(
        args.out / "summary.csv", index=False)
    print(f"results saved to {args.out}")


if __name__ == "__main__":
    main()
