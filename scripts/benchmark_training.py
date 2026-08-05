"""Run a short nnU-Net training loop without the expensive final validation.

This script is intended for deadline planning and memory checks, not for model
selection. Set the ``PANCREAS_MT_*`` environment variables before launching it.
"""

from __future__ import annotations

import argparse
import time

import torch
from nnunetv2.run.run_training import get_trainer_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="501")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--trainer", default="nnUNetTrainerPancreasMultiTask")
    parser.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    trainer = get_trainer_from_args(
        args.dataset,
        args.configuration,
        args.fold,
        args.trainer,
        args.plans,
        False,
        device,
    )
    trainer.disable_checkpointing = True
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    trainer.run_training()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"BENCHMARK_SECONDS={elapsed:.3f}")
    print(f"PEAK_ALLOCATED_MIB={torch.cuda.max_memory_allocated(device) / 1024**2:.1f}")
    print(f"PEAK_RESERVED_MIB={torch.cuda.max_memory_reserved(device) / 1024**2:.1f}")


if __name__ == "__main__":
    main()
