from __future__ import annotations

import argparse
import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.data import Cifar10DataModule
from src.lit_module import CifarLitModule
from src.model import cifar_resnet18, count_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # IO
    p.add_argument("--data_dir", type=str, required=True, help="CIFAR-10 root directory (will download if missing)")
    p.add_argument("--output_dir", type=str, required=True, help="Where to write logs/checkpoints")
    p.add_argument("--experiment", type=str, default="resnet18", help="Subdir name under output_dir")
    p.add_argument("--download", action="store_true", help="Download CIFAR-10 if not present (recommended)")

    # Train
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_size", type=int, default=5000, help="Validation size split from CIFAR-10 train set")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--precision", type=str, default="auto", help="auto | 16-mixed | 32-true")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="auto")  # auto/cpu/gpu

    # Optim
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    # Debug / speed
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--limit_test_batches", type=float, default=1.0)
    p.add_argument("--log_every_n_steps", type=int, default=50)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir).expanduser().resolve()
    exp_dir = output_dir / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)

    dm = Cifar10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=bool(args.download),
        val_size=int(args.val_size),
        seed=int(args.seed),
    )

    model = cifar_resnet18(num_classes=10)
    lit = CifarLitModule(
        model=model,
        num_classes=10,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        label_smoothing=args.label_smoothing,
    )

    logger = TensorBoardLogger(save_dir=str(exp_dir), name="tb")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(exp_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_last=True,
        save_top_k=1,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    precision = args.precision
    if precision == "auto":
        if args.accelerator == "cpu":
            precision = "32-true"
        elif args.accelerator == "gpu":
            precision = "16-mixed"
        else:
            precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    trainer = L.Trainer(
        default_root_dir=str(exp_dir),
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=precision,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=False,
        benchmark=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )

    print(f"[info] exp_dir: {exp_dir}")
    print(f"[info] model params: {count_params(model):,}")
    print(f"[info] accelerator: {trainer.accelerator.__class__.__name__}, devices: {args.devices}")
    print(f"[info] precision: {precision}")

    trainer.fit(lit, datamodule=dm)

    # Test with best checkpoint (if available)
    ckpt_path = ckpt_cb.best_model_path if ckpt_cb.best_model_path else "last"
    print(f"[info] testing with ckpt: {ckpt_path}")
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # Avoid tokenizer/parallelism noise on some environments
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

