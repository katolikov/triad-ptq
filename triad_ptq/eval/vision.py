"""ImageNet evaluation for CNNs / ViTs.

We use the ImageNetV2 'matched-frequency' subset (10K images) by default
because the official ImageNet val set requires manual download with
ToS acceptance. ImageNetV2 is downloaded from HuggingFace
`vaishaal/ImageNetV2` and the predictions are mapped through the
1000-class ImageNet schema using the standard ordering.
"""
from __future__ import annotations

import io
import time

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


def imagenet_default_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class HFImageNetV2(Dataset):
    """ImageNetV2 from HuggingFace `vaishaal/ImageNetV2`.

    The HF version ships three 10K subsets concatenated under the 'train'
    split. We select the matched-frequency subset (10K, the standard
    ImageNetV2 benchmark). Labels are encoded in the __key__ string:
    'imagenetv2-<subset>-format-val/<class_idx>/<hash>'.
    """

    DEFAULT_SUBSET = "matched-frequency-format-val"

    def __init__(self, transform=None, max_n: int | None = None,
                 subset: str = DEFAULT_SUBSET, seed: int = 0):
        from datasets import load_dataset
        ds = load_dataset("vaishaal/ImageNetV2", split="train")
        # Filter to the chosen subset using __key__ prefix.
        keep = [i for i, k in enumerate(ds["__key__"]) if subset in k]
        ds = ds.select(keep)
        if max_n is not None and max_n < len(ds):
            # Deterministic random subsample.
            import random
            r = random.Random(seed)
            idx = r.sample(range(len(ds)), max_n)
            idx.sort()
            ds = ds.select(idx)
        self.ds = ds
        self.transform = transform or imagenet_default_transform()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["jpeg"].convert("RGB")
        x = self.transform(img)
        # parse label from __key__: 'imagenetv2-.../987/abc...'
        y = int(item["__key__"].split("/")[-2])
        return x, y


@torch.no_grad()
def topk_accuracy(model, loader: DataLoader, *, device: torch.device,
                  k: tuple[int, ...] = (1, 5), progress: bool = True) -> dict:
    model.eval()
    n = 0
    correct = {kk: 0 for kk in k}
    t0 = time.perf_counter()
    iterator = tqdm(loader, desc="eval", leave=False) if progress else loader
    for x, y in iterator:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device)
        logits = model(x)
        if hasattr(logits, "logits"):
            logits = logits.logits
        topk = logits.topk(max(k), dim=-1).indices  # (N, max_k)
        for kk in k:
            correct[kk] += int((topk[:, :kk] == y.unsqueeze(1)).any(dim=1).sum().item())
        n += y.numel()
    return {
        "n": n,
        **{f"top{kk}": correct[kk] / max(n, 1) for kk in k},
        "sec": time.perf_counter() - t0,
    }


def build_imagenet_calibration(transform=None, n: int = 64, batch_size: int = 16,
                                seed: int = 0) -> list:
    """Build a small calibration set as a list of input batches."""
    ds = HFImageNetV2(transform=transform, max_n=max(n * 2, n), seed=seed)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    xs = [ds[i][0] for i in idx]
    batches = []
    for i in range(0, len(xs), batch_size):
        batches.append(torch.stack(xs[i : i + batch_size]))
    return batches
