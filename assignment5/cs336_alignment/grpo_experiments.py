import torch
import argparse
import json
from einops import rearrange
from typing import Callable, Literal
from torch.optim import Optimizer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from vllm_utils import VLLMServer, VLLMCompletion
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from checkpoint import get_model_and_tokenizer
import grpo
from utils import load_prompt, load_data
import wandb


def get_wandb_samples_table(step, completions, group_size, num_samples_to_log, repeated_prompts, repeated_ground_truths, table_name):
    wandb_data = {"step": step}
    columns = ["prompt", "response", "ground_truth"]
    table_data = []
    completions_text = [completion.text for completion in completions]
    for i in range(0, num_samples_to_log * group_size, group_size):
        table_data.append([
            repeated_prompts[i],
            completions_text[i],
            repeated_ground_truths[i],
        ])
    wandb_data[table_name] = wandb.Table(
        columns=columns,
        data=table_data,
    )
    return wandb_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--train_file_path", type=str, required=True)
    parser.add_argument("--val_file_path", type=str, required=True)

    # sampling hyperparams
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_max_tokens", type=int, default=512)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--rollout_batch_size", type=int, default=256)

    # training hyperparams
    parser.add_argument("--n_train_examples", type=int, default=6400)
    parser.add_argument("--num_rollout_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)

    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--n_val_examples", type=int, default=1024)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load data and prompt
    # train_set and val_set are a list of dicts with 'question' and 'answer'
    train_set = load_data(args.train_file_path)[: args.n_train_examples]
    val_set = load_data(args.val_file_path)
    prompt = load_prompt(args.prompt)

    # Init wandb
    group_name = f"grpo-{args.model.split('/')[-1]}-{args.prompt.split('/')[-1]}"
    wandb.init(
        project="cs336-a5-grpo",
        group=group_name,
        name=f"{group_name}-seed{args.seed}-lr{args.learning_rate}",
        config={
            "learning_rate": args.learning_rate,
            "rollout_batch_size": args.rollout_batch_size,
            "group_size": args.group_size,
            "sampling_temperature": args.sampling_temperature,
            "num_rollout_steps": args.num_rollout_steps,
            "seed": args.seed,
        },
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    policy, tokenizer = get_model_and_tokenizer(args.model, device)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )

    reward_fn = r1_zero_reward_fn

    server = VLLMServer(model_id=args.model, gpu=1, gpu_memory_utilization=0.4)

    server.start()
    server.init_weight_sync(policy_device=device)

    distinct_examples_per_step = args.rollout_batch_size // args.group_size
    N = len(train_set)

    # pass in None for seed to tell vLLM not to force each request through same deterministic stream
    sampling_params = {
        'temperature': args.sampling_temperature,
        'max_tokens': args.sampling_max_tokens,
        'n': 1,
        'seed': None,
        'stop': ["</answer>"],
        'include_stop_str_in_output': True,
    }

    for step in range(args.num_rollout_steps):
        server.sync_policy_weights(policy=policy)

        # get repeated_prompts, rollout_responses, and repeated_ground_truths
        # sample a batch from train set
        start_idx = (step * distinct_examples_per_step) % N
        rollout_indices = [(start_idx + i) % N for i in range(distinct_examples_per_step)]
        rollout_prompts = [train_set[i]['question'] for i in rollout_indices]
        rollout_ground_truths = [train_set[i]['answer'].split('####')[1].strip() for i in rollout_indices]

        # format + repeat prompts
        rollout_prompts = [prompt.format(question=ex) for ex in rollout_prompts]
        repeated_prompts = [p for p in rollout_prompts for _ in range(args.group_size)]
        repeated_ground_truths = [p for p in rollout_ground_truths for _ in range(args.group_size)]

        completions = server.generate_completions(
            prompts=repeated_prompts,
            sampling_params=sampling_params,
            batch_size=args.rollout_batch_size,
        )

        if step % 40 == 0:
            wandb_train_samples_table = get_wandb_samples_table(step, completions, args.group_size, 5, repeated_prompts, repeated_ground_truths, "train_samples")
            wandb.log(wandb_train_samples_table)

        batch_loss, metadata = grpo.grpo_train_step(
            model=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            reward_fn=reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=[completion.text for completion in completions],
            repeated_ground_truths=repeated_ground_truths,
            group_size=args.group_size,
            baseline="mean",
            advantage_eps=args.advantage_eps,
            advantage_normalizer="std",
            importance_reweighting_method="none",
            old_log_probs=None,
            cliprange=None,
            loss_normalization="sequence",
            normalization_constant=None,
        )

        wandb_data = {
            "step": step,
            "batch_loss": batch_loss,
        }
        wandb_data = wandb_data | metadata
        wandb.log(wandb_data)

        if step % args.eval_every == 0:
            val_rollout_prompts = [val_set[i]['question'] for i in range(args.n_val_examples)]
            val_rollout_ground_truths = [val_set[i]['answer'].split('####')[1].strip() for i in range(args.n_val_examples)]
            val_rollout_prompts = [prompt.format(question=ex) for ex in val_rollout_prompts]
            val_repeated_prompts = [p for p in val_rollout_prompts for _ in range(args.group_size)]
            val_repeated_ground_truths = [p for p in val_rollout_ground_truths for _ in range(args.group_size)]
            val_completions = server.generate_completions(
                prompts=val_repeated_prompts,
                sampling_params=sampling_params,
                batch_size=None,
            )
            completions_text = [completion.text for completion in val_completions]
            rewards_tensor, rewards_metadata = grpo.compute_rollout_rewards(
                reward_fn=reward_fn, rollout_responses=completions_text, repeated_ground_truths=val_repeated_ground_truths
            )
            tokenized_completions = tokenizer(
                completions_text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]

            avg_response_length = sum(
                len(token_ids) for token_ids in tokenized_completions
            ) / len(tokenized_completions)

            wandb_data = {
                "step": step,
                "val/avg_response_length": avg_response_length,
            }
            for k, v in rewards_metadata.items():
                new_k = f"val/{k}"
                wandb_data[new_k] = v

            wandb_val_samples_table = get_wandb_samples_table(step, val_completions, args.group_size, 5, val_repeated_prompts, val_repeated_ground_truths, "val_samples")
            wandb.log(wandb_data | wandb_val_samples_table)


    server.stop()


if __name__ == "__main__":
    main()
