import pickle
import numpy as np
import torch
from tokenizer import Tokenizer
from transformer import TransformerLM
from utils import load_checkpoint, decode
from training_loop import train

vocab_size, context_length, d_model, d_ff, rope_theta, num_layers, num_heads = None, None, None, None, None, None, None

DEFAULT = True

if DEFAULT:
    vocab_size = 10000
    context_length = 256
    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    num_layers = 4
    num_heads = 16


def prepare_dataset(text_path, pkl_path, out_path):
    with open(pkl_path, 'rb') as f:
        bpe = pickle.load(f)
    tokenizer = Tokenizer(vocab=bpe['vocab'], merges=bpe['merges'], special_tokens=["<|endoftext|>"])

    with open(text_path, 'r') as f:
        text = f.read(1_000_000)

    tokens = tokenizer.encode(text)
    arr = np.array(tokens, dtype=np.int32)
    np.save(out_path, arr)
    print(f"Saved {len(arr)} tokens to {out_path}", flush=True)



def run_experiment():
    prepare_dataset(
        text_path="../data/tinystories.txt",
        pkl_path="../tinystories-train-bpe.pkl",
        out_path="../tinystories_tokens.npy"
    )

    dataset = np.load("../tinystories_tokens.npy")
    val_dataset = np.load("../tinystories_val_tokens.npy")

    # with open('../tinystories-train-bpe.pkl', 'rb') as f:
    #     bpe = pickle.load(f)

    batch_sz = 256
    lr = 8e-3

    train(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
        rope_theta=rope_theta,
        learning_rate=lr,
        lr_min=1e-4,
        warmup_iters=100,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        max_l2_norm=1.0,
        num_iterations=5000,
        save_every=500,
        dataset=dataset,
        val_set=val_dataset,
        batch_size=batch_sz,
        device="cpu",
        ckpt_path=None,
        out_path=f"../experiments_bs_{batch_sz}_max_lr_{lr}/tinystories_ckpt.pt",
        resume=False,
        run_name=f"batchsz_{batch_sz}_max_lr_{lr}",
    )

def generate(ckpt_path, prompt_text, max_tokens=200, temperature=1.0, top_p=0.9):
    with open('../tinystories-train-bpe.pkl', 'rb') as f:
        bpe = pickle.load(f)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    load_checkpoint(ckpt_path, model, optimizer)
    model.eval()

    tokenizer = Tokenizer(vocab=bpe['vocab'], merges=bpe['merges'], special_tokens=["<|endoftext|>"])
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt = torch.tensor([prompt_tokens], dtype=torch.long)

    output = decode(model, prompt, bpe['vocab'], bpe['merges'], ["<|endoftext|>"], max_tokens, temperature, top_p)
    print(output)


if __name__ == '__main__':
    # run_experiment()
    generate("../experiments_max_lr_0.001/tinystories_ckpt.pt", prompt_text="Once upon a time", max_tokens=300, temperature=0.8, top_p=0.9)



