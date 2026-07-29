import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math
import argparse
import numpy as np
import os
from transformer import TransformerLM
from utils import cross_entropy, AdamW, cosine_lr_schedule, gradient_clipping, load_data, save_checkpoint, load_checkpoint
import wandb


def train(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    num_heads: int,
    d_model: int,
    d_ff: int,
    rope_theta: float,

    learning_rate: float,
    lr_min: float,
    warmup_iters: int,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,

    max_l2_norm: float,

    num_iterations: int,
    save_every: int,
    dataset: np.ndarray,
    val_set: np.ndarray,
    batch_size: int,
    device: str,

    ckpt_path: str,
    out_path: str,
    resume: bool,
    run_name: str = None,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
        rope_theta=rope_theta
    )
    optimizer = AdamW(params=model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)

    model = model.to(device)

    wandb.init(
        project="cs336-a1",
        name=run_name,
        config={
            "vocab_size": vocab_size,
            "num_heads": num_heads,
            "d_model": d_model,
            "d_ff": d_ff,
            "rope_theta": rope_theta,
            "context_length": context_length,
            "num_layers": num_layers,
            "lr": learning_rate,
            "lr_min": lr_min,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "max_l2_norm": max_l2_norm,
            "betas": betas,
            "eps": eps,
        },
    )

    start_iter = 0
    if resume:
        ckpt_iter = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Loaded from checkpoint at iteration {ckpt_iter}")

        start_iter = ckpt_iter + 1

    model.train()

    for i in range(start_iter, num_iterations):
        inputs, outputs = load_data(dataset=dataset, batch_size=batch_size, context_length=context_length, device=device)
        optimizer.zero_grad()
        y = model(inputs)
        loss = cross_entropy(y, outputs)
        loss.backward()

        gradient_clipping(model.parameters(), max_l2_norm)

        lr = cosine_lr_schedule(i, learning_rate, lr_min, warmup_iters, num_iterations)
        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.step()

        perplexity = torch.exp(loss)
        print(f"Iteration {i}: Loss: {loss.item():.4f}, Perplexity: {perplexity.item():.4f}")

        wandb_data = {
            "train/loss": loss.item(),
            "train/perplexity": perplexity.item(),
            "lr": lr,
        }

        if i % save_every == 0:
            out_file = out_path.replace(".pt", f"_iter{i}.pt")
            save_checkpoint(model, optimizer, i, out_file)
            print(f"Checkpoint at iteration {i} saved to {out_file}")
        
        if i % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_outputs = load_data(dataset=val_set, batch_size=batch_size, context_length=context_length, device=device)
                val_loss = cross_entropy(model(val_inputs), val_outputs)
            val_perplexity = torch.exp(val_loss)
            wandb_data["val/loss"] = val_loss.item()
            wandb_data["val/perplexity"] = val_perplexity.item()
            print(f"Validation Loss: {val_loss.item():.4f}, Validation Perplexity: {val_perplexity.item():.4f}")
            model.train()

        wandb.log(wandb_data)

    save_checkpoint(model, optimizer, num_iterations - 1, out_path)
    print(f"Final checkpoint saved to {out_path}")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1024)

    parser.add_argument("--rope_theta", type=float, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--lr_min", type=float, default=0.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_l2_norm", type=float, required=True)

    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_path", type=str, required=False)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    dataset = np.load(args.in_file)

    train(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        learning_rate=args.learning_rate,
        lr_min=args.lr_min,
        warmup_iters=args.warmup_iters,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
        max_l2_norm=args.max_l2_norm,
        num_iterations=args.num_iterations,
        save_every=args.save_every,
        dataset=dataset,
        batch_size=args.batch_size,
        device=args.device,
        ckpt_path=args.ckpt_path,
        out_path=args.out_path,
        resume=args.resume,
    )


if __name__ == '__main__':
    main()

