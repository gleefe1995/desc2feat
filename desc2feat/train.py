"""Train Desc2Feat with geometry supervision; supports CPU smoke tests and torchrun."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .config import load_config
from .data import PairDataset, SyntheticPairs, collate_pairs, to_device
from .losses import Desc2FeatLoss
from .model import Desc2Feat


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _save(path, model, config, optimizer, scheduler, scaler, epoch, batch_in_epoch, step, rank, world):
    states = [None] * world if rank == 0 else None
    state = rng_state()
    if world > 1:
        dist.gather_object(state, states, dst=0)
    else:
        states = [state]
    if rank == 0:
        raw = model.module if hasattr(model, "module") else model
        payload = {"format_version": 1, "model": raw.state_dict(), "config": config,
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "scaler": scaler.state_dict(), "epoch": epoch, "batch_in_epoch": batch_in_epoch,
                   "step": step, "rng_states": states, "world_size": world}
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)


def _autocast(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--output", default="runs/desc2feat")
    parser.add_argument("--resume", help="Resume a trusted training checkpoint, including optimizer and RNG")
    parser.add_argument("--eloftr-checkpoint", help="Initialize the compatible eLoFTR backbone")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-steps", type=int, help="Maximum total optimizer updates (also when resuming)")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.eloftr_checkpoint:
        parser.error("--resume and --eloftr-checkpoint cannot be combined")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    config_path = args.config
    if args.smoke_test and not config_path and not checkpoint:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "smoke.json"
    cfg = checkpoint["config"] if checkpoint and not args.config else load_config(config_path)
    train, data = cfg.setdefault("train", {}), cfg.setdefault("data", {})
    defaults = {"epochs": 30, "batch_size": 2, "lr": 1e-4, "weight_decay": 1e-4,
                "accumulation_steps": 1, "amp": True, "grad_clip": 1., "seed": 42,
                "log_every": 20, "save_every": 1}
    for key, value in defaults.items():
        train.setdefault(key, value)
    for key, value in {"resize": 832, "pad_multiple": 32, "workers": 4}.items():
        data.setdefault(key, value)
    for key in ("epochs", "batch_size"):
        if getattr(args, key) is not None:
            train[key] = getattr(args, key)
    if args.workers is not None:
        data["workers"] = args.workers
    if args.train_manifest:
        data["train_manifest"] = args.train_manifest
    if args.val_manifest:
        data["val_manifest"] = args.val_manifest
    if args.smoke_test:
        data["workers"] = 0
        train["batch_size"] = args.batch_size or 1
        train["epochs"] = args.epochs or 1
        train["log_every"] = 1
        args.max_steps = args.max_steps or 2
    if any(train[key] < 1 for key in ("epochs", "batch_size", "accumulation_steps", "log_every", "save_every")):
        parser.error("epochs, batch_size, accumulation_steps, log_every, and save_every must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max_steps must be positive")
    world, rank, local_rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(args.device)
    if world > 1:
        if device.type == "cuda":
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    if checkpoint and checkpoint.get("world_size", 1) != world:
        raise ValueError("Exact resume requires the same WORLD_SIZE as the checkpoint.")
    seed_everything(int(train["seed"]) + rank)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        dataset = SyntheticPairs(length=max(4, train["batch_size"] * world * train["accumulation_steps"] * 2), size=64, seed=train["seed"])
        validation = SyntheticPairs(length=2, size=64, seed=train["seed"] + 10000)
    else:
        if not data.get("train_manifest"):
            parser.error("provide --train-manifest or use --smoke-test")
        dataset = PairDataset(data["train_manifest"], data["resize"])
        validation = PairDataset(data["val_manifest"], data["resize"]) if data.get("val_manifest") else None
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                 seed=train["seed"], drop_last=False)
    loader_generator = torch.Generator()
    collate = partial(collate_pairs, pad_multiple=data["pad_multiple"])
    loader = DataLoader(dataset, batch_size=train["batch_size"], sampler=sampler,
                        num_workers=data["workers"], collate_fn=collate, pin_memory=device.type == "cuda",
                        generator=loader_generator, persistent_workers=False)
    if validation is not None:
        validation = Subset(validation, list(range(rank, len(validation), world)))
        val_loader = DataLoader(validation, batch_size=train["batch_size"], num_workers=data["workers"],
                                collate_fn=collate, pin_memory=device.type == "cuda")
    else:
        val_loader = None
    model = Desc2Feat(cfg["model"]).to(device)
    if args.eloftr_checkpoint:
        report = model.load_eloftr_backbone(args.eloftr_checkpoint)
        if rank == 0:
            print(json.dumps({"eloftr_initialization": report}, default=str), flush=True)
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    criterion = Desc2FeatLoss(cfg["loss"]).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                                        find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["lr"], weight_decay=train["weight_decay"])
    updates_per_epoch = math.ceil(len(loader) / train["accumulation_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, updates_per_epoch * train["epochs"]))
    amp_enabled = bool(train["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, start_batch, step = 0, 0, 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, start_batch, step = checkpoint["epoch"], checkpoint.get("batch_in_epoch", 0), checkpoint["step"]
        restore_rng(checkpoint["rng_states"][rank])
    if rank == 0:
        (output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(json.dumps({"device": str(device), "world_size": world, "train_pairs": len(dataset),
                          "parameters": sum(p.numel() for p in model.parameters()), "resume_step": step,
                          "synthetic_smoke_test": args.smoke_test}), flush=True)
    last_checkpoint = Path(args.resume) if args.resume else None
    stop = args.max_steps is not None and step >= args.max_steps
    started = time.perf_counter()
    for epoch in range(start_epoch, train["epochs"]):
        if stop:
            break
        model.train()
        sampler.set_epoch(epoch)
        loader_generator.manual_seed(train["seed"] + epoch)
        optimizer.zero_grad(set_to_none=True)
        consumed = start_batch if epoch == start_epoch else 0
        for batch_index, batch in enumerate(loader):
            if batch_index < consumed:
                continue
            batch = to_device(batch, device)
            accumulation = train["accumulation_steps"]
            window_start = batch_index // accumulation * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            update_now = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            sync = model.no_sync() if world > 1 and not update_now else nullcontext()
            with sync:
                with _autocast(device, amp_enabled):
                    result = model(batch)
                    terms = criterion(result, batch)
                    loss = terms["loss"] / window_size
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
                scaler.scale(loss).backward()
            consumed = batch_index + 1
            if not update_now:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"],
                                                        error_if_nonfinite=not amp_enabled)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            did_update = scaler.get_scale() >= scale_before
            if did_update:
                scheduler.step()
                step += 1
            optimizer.zero_grad(set_to_none=True)
            if rank == 0 and (step % train["log_every"] == 0 or step == 1):
                record = {"split": "train", "epoch": epoch, "step": step, "batch": batch_index,
                          "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm),
                          "elapsed_s": time.perf_counter() - started,
                          **{key: float(value.detach()) for key, value in terms.items() if isinstance(value, torch.Tensor) and value.numel() == 1}}
                print(json.dumps(record), flush=True)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
        if val_loader is not None:
            validation_rng = rng_state()
            raw = model.module if hasattr(model, "module") else model
            raw.eval()
            if world > 1:
                # Use the checkpoint rank's BN statistics consistently on every validation shard.
                for buffer in raw.buffers():
                    dist.broadcast(buffer, src=0)
            total = torch.zeros(2, device=device)
            with torch.no_grad():
                for batch in val_loader:
                    batch = to_device(batch, device)
                    with _autocast(device, amp_enabled):
                        result = raw(batch, return_training_outputs=True)
                        value = criterion(result, batch)["loss"]
                    count = batch["image0"].shape[0]
                    total += torch.stack((value.float() * count, torch.tensor(float(count), device=device)))
            restore_rng(validation_rng)
            if world > 1:
                dist.all_reduce(total)
            if rank == 0:
                record = {"split": "validation", "epoch": epoch, "step": step,
                          "loss": float(total[0] / total[1].clamp_min(1)), "pairs": int(total[1])}
                print(json.dumps(record), flush=True)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
        next_epoch, next_batch = (epoch + 1, 0) if consumed >= len(loader) else (epoch, consumed)
        _save(output / "last.pt", model, cfg, optimizer, scheduler, scaler,
              next_epoch, next_batch, step, rank, world)
        last_checkpoint = output / "last.pt"
        if (epoch + 1) % train["save_every"] == 0 and consumed >= len(loader):
            _save(output / f"epoch-{epoch + 1:04d}.pt", model, cfg, optimizer, scheduler, scaler,
                  next_epoch, next_batch, step, rank, world)
        start_batch = 0
    if rank == 0:
        print(json.dumps({"finished": True, "step": step, "checkpoint": str(last_checkpoint) if last_checkpoint else None}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
