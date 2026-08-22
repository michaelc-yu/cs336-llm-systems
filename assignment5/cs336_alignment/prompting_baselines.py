
import json
from vllm_utils import VLLMServer, VLLMCompletion
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from utils import load_prompt, load_data


def grade(responses, ground_truths, reward_fn):
    rewards = [
        reward_fn(response.text, truth) for response, truth in zip(responses, ground_truths)
    ]
    return rewards

def print_reward_categories(name, rewards):
    both_correct = sum(r["format_reward"] == 1 and r["answer_reward"] == 1 for r in rewards)
    format_only = sum(r["format_reward"] == 1 and r["answer_reward"] == 0 for r in rewards)
    neither = sum(r["format_reward"] == 0 and r["answer_reward"] == 0 for r in rewards)
    print(f"{name}")
    print(f"Format=1, Correctness=1: {both_correct}")
    print(f"Format=1, Correctness=0: {format_only}")
    print(f"Format=0, Correctness=0: {neither}")


def main():
    examples = load_data('data/gsm8k/test.jsonl')

    total_examples = len(examples)

    ground_truths = [ex["answer"].split("####")[1].strip() for ex in examples]

    question_only_prompt = load_prompt('cs336_alignment/prompts/question_only.prompt')

    r1_zero_three_shot_prompt = load_prompt('cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt')

    r1_zero_prompt = load_prompt('cs336_alignment/prompts/r1_zero.prompt')

    prompts = {
        "question_only": [],
        "r1_zero": [],
        "r1_zero_three_shot": [],
    }

    for ex in examples:
        prompts["question_only"].append(
            question_only_prompt.format(question=ex["question"])
        )
        prompts["r1_zero"].append(
            r1_zero_prompt.format(question=ex["question"])
        )
        prompts["r1_zero_three_shot"].append(
            r1_zero_three_shot_prompt.format(question=ex["question"])
        )

    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=0)

    server.start()

    sampling_params = {
        'temperature': 1.0,
        'max_tokens': 512,
        'n': 1,
        'seed': 42,
        'stop': ["</answer>"],
        'include_stop_str_in_output': True,
    }

    completions = server.generate_completions(
        prompts=prompts['question_only'] + prompts['r1_zero'] + prompts['r1_zero_three_shot'],
        sampling_params=sampling_params,
        batch_size=32,
    )

    # print(completions[0])

    server.stop()

    question_only_completions = completions[: total_examples]
    r1_zero_completions = completions[total_examples : total_examples * 2]
    r1_zero_three_shot_completions = completions[total_examples * 2 : total_examples * 3]

    assert len(question_only_completions) == len(r1_zero_completions) == len(r1_zero_three_shot_completions) == total_examples

    question_only_rewards = grade(question_only_completions, ground_truths, question_only_reward_fn)
    r1_zero_rewards = grade(r1_zero_completions, ground_truths, r1_zero_reward_fn)
    r1_zero_three_shot_rewards = grade(r1_zero_three_shot_completions, ground_truths, r1_zero_reward_fn)

    print_reward_categories("Question-only", question_only_rewards)
    print_reward_categories("R1-zero", r1_zero_rewards)
    print_reward_categories("R1-zero 3-shot", r1_zero_three_shot_rewards)

if __name__ == "__main__":
    main()

