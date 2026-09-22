import torch
from einops import rearrange
from typing import Callable, Literal
from torch.optim import Optimizer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False, padding=False, truncation=True)["input_ids"]
    output_ids = tokenizer(output_strs, add_special_tokens=False, padding=False, truncation=True)["input_ids"]

    assert len(prompt_ids) == len(output_ids)

    input_ids = [prompt + response for prompt, response in zip(prompt_ids, output_ids)]

    # Pad the sequences
    max_len = max(len(ids) for ids in input_ids)
    padded_ids = []
    for id in input_ids:
        pad_len = max_len - len(id)
        padded_ids.append(id + [tokenizer.pad_token_id] * pad_len)

    input_ids = torch.tensor(padded_ids)

    # Construct mask
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, (prompt, output) in enumerate(zip(prompt_ids, output_ids)):
        len_prompt = len(prompt)
        len_output = len(output)
        response_mask[i, len_prompt : len_prompt + len_output] = 1


    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    response_mask = response_mask[:, 1:]

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # shape (batch_sz, seq_len, vocab_size)
    token_log_probs = rearrange(
        torch.gather(log_probs, 2, rearrange(labels, "b s -> b s 1")),
        "b s 1 -> b s",
    )

    res = {'log_probs': token_log_probs}
    if return_token_entropy:
        token_entropy = -torch.sum(log_probs * torch.exp(log_probs), dim=-1)
        res['token_entropy'] = token_entropy

    return res


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    assert len(rollout_responses) == len(repeated_ground_truths)

    all_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        # reward_dict contains keys 'format_reward', 'answer_reward', and 'reward'
        reward_dict = reward_fn(response, ground_truth)
        all_rewards.append(reward_dict)

    raw_rewards = torch.stack([torch.tensor(reward_dict['reward']) for reward_dict in all_rewards])

    mean_reward = sum(reward['reward'] for reward in all_rewards) / len(all_rewards)
    mean_format_reward = sum(reward['format_reward'] for reward in all_rewards) / len(all_rewards)
    metadata = {
        'mean_reward': mean_reward,
        'mean_format_reward': mean_format_reward,
    }

    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal['mean', 'none'] = 'mean',
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal['std', 'none', 'mean'] = 'std',
):
    rewards = rearrange(raw_rewards, "(g n) -> g n", n = group_size)

    if baseline == 'mean':
        group_means = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - group_means
    elif baseline == 'none':
        advantages = rewards
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    if advantage_normalizer == 'std':
        group_stds = rewards.std(dim=1, keepdim=True) + advantage_eps
        normalized_advantages = advantages / group_stds
    elif advantage_normalizer == 'mean':
        group_means = rewards.mean(dim=1, keepdim=True) + advantage_eps
        normalized_advantages = advantages / group_means
    elif advantage_normalizer == 'none':
        normalized_advantages = advantages
    else:
        raise ValueError(f"Unknown advantage normalizer: {advantage_normalizer}")

    metadata = {
        'mean': raw_rewards.mean().item(),
        'std': raw_rewards.std().item(),
        'min': raw_rewards.min().item(),
        'max': raw_rewards.max().item(),
    }
    normalized_advantages = rearrange(normalized_advantages, "g n -> (g n)", n = group_size)
    return normalized_advantages, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal['none', 'noclip', 'grpo', 'gspo'] = 'none',
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    metadata = {}
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = rearrange(raw_rewards_or_advantages, "b -> b 1",)
    
    if importance_reweighting_method == 'none':
        per_token_policy_gradient_loss = -raw_rewards_or_advantages * policy_log_probs
    else:
        raise NotImplementedError

    if response_mask is not None:
        per_token_policy_gradient_loss = per_token_policy_gradient_loss * response_mask

    return per_token_policy_gradient_loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal['sequence', 'constant'] = 'sequence',
    normalization_constant: int | None = None,
    normalization_count: int | None = None,
) -> torch.Tensor:
    masked_loss = per_token_policy_gradient_loss * mask
    per_sequence_loss_sum = masked_loss.sum(dim=1)

    if loss_normalization == 'sequence':
        num_response_tokens = mask.sum(dim=1)
        num_response_tokens = torch.clamp(num_response_tokens, min=1)
        per_sequence_loss = per_sequence_loss_sum / num_response_tokens
        if normalization_count is None:
            scalar_loss = per_sequence_loss.mean()
        else:
            scalar_loss = per_sequence_loss.sum() / normalization_count
    elif loss_normalization == 'constant':
        assert normalization_constant is not None, "Normalization constant must be provided for 'constant' normalization."
        sequence_loss_sum = per_sequence_loss_sum.sum()
        scalar_loss = sequence_loss_sum / normalization_constant
    else:
        raise ValueError(f"Unknown loss normalization: {loss_normalization}")

    return scalar_loss


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "mean", "none"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    
    # Compute rewards
    rewards_tensor, rewards_metadata = compute_rollout_rewards(
        reward_fn=reward_fn, rollout_responses=rollout_responses, repeated_ground_truths=repeated_ground_truths
    )

    # Compute advantages
    advantages, advantages_metadata = compute_group_normalized_rewards(
        raw_rewards=rewards_tensor,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    batch_loss = 0
    all_loss_metadata = []

    microbatch_size = len(repeated_prompts) // gradient_accumulation_steps
    for i in range(0, len(repeated_prompts), microbatch_size):
        inputs_microbatch = repeated_prompts[i : i + microbatch_size]
        labels_microbatch = rollout_responses[i : i + microbatch_size]
        advantages_microbatch = advantages[i : i + microbatch_size]

        # filter out zero-advantage sequences
        original_microbatch_size = len(inputs_microbatch)
        nonzero_mask = advantages_microbatch != 0
        if not nonzero_mask.any():
            continue
        inputs_microbatch = [
            x for x, keep in zip(inputs_microbatch, nonzero_mask) if keep
        ]
        labels_microbatch = [
            x for x, keep in zip(labels_microbatch, nonzero_mask) if keep
        ]
        advantages_microbatch = advantages_microbatch[nonzero_mask]

        # Tokenize and get log probs
        output = tokenize_prompt_and_output(
            prompt_strs=inputs_microbatch,
            output_strs=labels_microbatch,
            tokenizer=tokenizer,
        )
        response_mask = output["response_mask"].to(model.device)

        current_log_probs_dict = get_response_log_probs(
            model=model,
            input_ids=output['input_ids'].to(model.device),
            labels=output['labels'].to(model.device),
            return_token_entropy=True,
        )
        current_token_log_probs = current_log_probs_dict['log_probs']
        current_token_entropy = current_log_probs_dict["token_entropy"]

        mean_response_entropy = (
            (current_token_entropy * response_mask).sum()
            / response_mask.sum().clamp(min=1)
        )

        advantages_microbatch = advantages_microbatch.to(model.device)

        # Compute per-token loss
        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=advantages_microbatch,
            policy_log_probs=current_token_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
            response_mask=response_mask,
        )
        loss_metadata['mean_token_entropy'] = mean_response_entropy.detach().item()
        all_loss_metadata.append(loss_metadata)

        # Aggregate to scalar
        scalar_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=output['response_mask'].to(model.device),
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
            normalization_count=original_microbatch_size,
        )
        if loss_normalization == 'sequence':
            scalar_loss /= gradient_accumulation_steps
        scalar_loss.backward()
        batch_loss += scalar_loss

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    optimizer.step()

    optimizer.zero_grad()

    all_loss_metadata_dict = {}
    if all_loss_metadata:
        for k in all_loss_metadata[0].keys():
            values = [metadata[k] for metadata in all_loss_metadata]
            all_loss_metadata_dict[k] = sum(values) / len(values)

    metadata = rewards_metadata | advantages_metadata
    metadata = metadata | all_loss_metadata_dict

    if max_grad_norm is not None:
        metadata['grad_norm'] = grad_norm

    return batch_loss, metadata
