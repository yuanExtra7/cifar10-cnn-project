from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch

from src.data import Cifar10DataModule
from src.lit_module import CifarLitModule
from src.model import cifar_resnet18


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="CIFAR-10 root directory")
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to a .ckpt checkpoint file")
    p.add_argument("--out_json", type=str, default="reports/test_metrics.json", help="Where to write metrics json")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--download", action="store_true", help="Download CIFAR-10 if not present")
    p.add_argument("--accelerator", type=str, default="auto")  # auto/cpu/gpu
    p.add_argument("--devices", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    dm = Cifar10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=bool(args.download),
    )

    model = cifar_resnet18(num_classes=10)
    lit = CifarLitModule.load_from_checkpoint(str(ckpt_path), model=model)

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision="16-mixed" if torch.cuda.is_available() and args.accelerator != "cpu" else "32-true",
        logger=False,
        enable_checkpointing=False,
    )

    results = trainer.test(lit, datamodule=dm)
    metrics = results[0] if results else {}

    payload = {
        "ckpt_path": str(ckpt_path),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "metrics": metrics,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()

