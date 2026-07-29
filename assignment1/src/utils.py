import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math
import numpy as np
from tokenizer import Tokenizer
from layers import softmax


def cross_entropy(logits, targets):
    """
    Compute the cross-entropy loss between logits and targets.
    """
    # for numerical stability
    largest = torch.max(logits, dim=-1, keepdim=True).values
    logits = logits - largest

    # get the logits corresponding to the target tokens
    target_logits = torch.gather(
        logits,
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    # negative log likelihood
    # likelihood is just the softmax of the target logits, which is exp(target_logits) / sum(exp(logits))
    # log(a/b) = log(a) - log(b)
    # -log(a/b) = -(log(a) - log(b)) = log(b) - log(a)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))
    return torch.mean(log_sum_exp - target_logits)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0 or betas[0] < 0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                grad = p.grad.data

                t = state.get('t', 1)
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * (grad * grad)
                
                adjusted_lr = lr * (math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t))
                p.data -= adjusted_lr * (m / torch.sqrt(v + eps))
                p.data -= lr * weight_decay * p.data

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
        return loss


def cosine_lr_schedule(t, lr_max, lr_min, T_w, T_c):
    """
    Compute the learning rate at time step t using a cosine schedule with warmup.
    """
    if t < T_w:
        return lr_max * t / T_w
    elif T_w <= t <= T_c:
        return lr_min + 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (lr_max - lr_min)
    else:
        return lr_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6

    grad = [p.grad for p in parameters if p.grad is not None]
    grads_flat = torch.cat([g.detach().flatten() for g in grad])
    
    l2_norm = torch.norm(grads_flat, 2)
    if l2_norm < max_l2_norm:
        return l2_norm
    else:
        factor = max_l2_norm / (l2_norm + eps)
        for g in grad:
            g *= factor


def load_data(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    sz = len(dataset)
    max_start_idx = sz - context_length - 1
    start_idx = np.random.randint(0, max_start_idx + 1, size=batch_size)

    x = np.stack([dataset[i:i+context_length] for i in start_idx])
    y = np.stack([dataset[i+1:i+context_length+1] for i in start_idx])

    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    obj = {}
    obj['model_state_dict'] = model.state_dict()
    obj['optimizer_state_dict'] = optimizer.state_dict()
    obj['iteration'] = iteration
    torch.save(obj, out)

def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    states = torch.load(src)
    model.load_state_dict(states['model_state_dict'])
    optimizer.load_state_dict(states['optimizer_state_dict'])
    return states["iteration"]


def decode(model, prompt: torch.Tensor, vocab, merges, special_tokens, max_tokens: int, temperature: float, top_p_threshold: float):
    generated_tokens = []
    
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    endoftext_token = tokenizer.encode("<|endoftext|>")[0]

    with torch.no_grad():
        while len(generated_tokens) < max_tokens:
            if generated_tokens and generated_tokens[-1] == endoftext_token:
                break
            # out is tensor of logits
            out = model(prompt)

            # scale the logits by temperature before turning them into probabilities
            out = out / temperature

            probs = softmax(out, dim=-1)[0, -1]

            # top-p sampling
            sorted_probs, indices = torch.sort(probs, descending=True)
            accumulated = 0
            p = len(vocab)
            for i in range(len(sorted_probs)):
                accumulated += sorted_probs[i]
                if accumulated >= top_p_threshold:
                    p = i + 1
                    break
            
            # keep only nucleus tokens
            nucleus_probs = sorted_probs[:p]
            nucleus_indices = indices[:p]

            # renormalize
            final_probs = nucleus_probs / nucleus_probs.sum()

            # sample
            sampled = torch.multinomial(final_probs, num_samples=1)
            next_token = nucleus_indices[sampled].item()

            generated_tokens.append(next_token)
            prompt = torch.cat([prompt, torch.tensor([[next_token]])], dim=1)

    return tokenizer.decode(generated_tokens)

