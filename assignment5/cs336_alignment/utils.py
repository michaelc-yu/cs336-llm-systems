
import json

def load_data(filepath: str):
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def load_prompt(prompt_path: str):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    return prompt

