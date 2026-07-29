import os
from src.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
import regex as re
import collections
from collections import Counter
from tqdm import tqdm


NUM_CHUNKS = 64
NUM_PROCESSES = 8

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_chunk(input_path, start, end, special_tokens):
    escaped_special_tokens = list(map(re.escape, special_tokens))
    special_tokens_regex = "|".join(escaped_special_tokens)

    chunk_pretoken_freqs = collections.defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        chunk = chunk.decode("utf-8", errors="ignore")
    
    docs = re.split(special_tokens_regex, chunk)
    for doc in docs:
        for pretoken in re.finditer(PAT, doc):
            pretoken_bytes = pretoken.group().encode("utf-8")
            chunk_pretoken_freqs[pretoken_bytes] += 1
    return chunk_pretoken_freqs

def pretokenize_chunk_wrapper(args):
    (input_path, start, end, special_tokens) = args
    return pretokenize_chunk(input_path, start, end, special_tokens)

def get_pair_to_merge(pair_freqs):
    return max(pair_freqs, key=lambda p: (pair_freqs[p], p))

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, NUM_CHUNKS, "<|endoftext|>".encode("utf-8"))
    
    # print(f"[DBG] chunk_boundaries: {chunk_boundaries}")

    # Pretokenization step first:
    # preprocessing text to split it into sensible chunks for your tokenizer to work on.
    # parallel pretokenize chunks
    args_list = []
    chunks = zip(chunk_boundaries[:-1], chunk_boundaries[1:])
    for (start, end) in chunks:
        args_list.append((input_path, start, end, special_tokens))

    # pretoken_id -> frequency
    pretoken_freqs = collections.defaultdict(int)
    # pretoken_id -> subwords
    pretoken_subwords = {}

    # Maps each pair -> which pretoken & which positions it occurs in. This lets you only touch the minimal set of affected pretokens when a pair is merged.
    # (b'i', b'r'): {0: {0}, 37: {2}} means 'ir' pair occurs
    # in first pretoken at index 0 and 38th pretoken at index 2
    pair_positions = collections.defaultdict(lambda: collections.defaultdict(set))

    # Keeps track of how often each pair appears, across all pretokens weighted by frequency
    pair_freqs = Counter()

    pool = Pool(NUM_PROCESSES)
    results = pool.map(pretokenize_chunk_wrapper, args_list)
    pool.close()
    pool.join()

    # merge matching byte frequencies across chunks safely
    global_raw_freqs = Counter()
    for chunk_pretoken_freqs in results:
        for pretoken, freq in chunk_pretoken_freqs.items():
            global_raw_freqs[pretoken] += freq

    # use a flat set for positions to bypass shifting-index bugs
    pretoken_freqs = {}
    pretoken_subwords = {}
    pair_positions = collections.defaultdict(set)
    pair_freqs = Counter()

    # assign a guaranteed, non-resetting unique ID to every word sequence
    for pretoken_id, (pretoken, freq) in enumerate(global_raw_freqs.items()):
        pretoken_freqs[pretoken_id] = freq
        subwords = [bytes([b]) for b in pretoken]
        pretoken_subwords[pretoken_id] = subwords

        for pair in zip(subwords[:-1], subwords[1:]):
            pair_positions[pair].add(pretoken_id)
            pair_freqs[pair] += freq

    # print(f"[DBG] pretoken freqs:\n{pretoken_freqs}")
    # print(f"[DBG] pretoken subwords:\n{pretoken_subwords}")
    # print(f"[DBG] pair_positions:\n{pair_positions}")
    # print(f"[DBG] pair_freqs:\n{pair_freqs}")

    # tokenizer vocabulary: a mapping from int (token ID in the vocabulary) to bytes (token bytes)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    # print(f"[DBG] vocab: {vocab}")

    # merges: list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>),
    # representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    merges: list[tuple[bytes, bytes]] = []


    # the bpe example code is slow because for every merge, it iterates over all byte
    # pairs to identify the most frequent pair. However, the only pair counts that
    # change after each merge are those that overlap with the merged pair
    # thus, bpe training speed can be improved by indexing the counts of all pairs
    # and incrementally updating these counts, rather than explicitly iterating over
    # each pair of bytes to count pair frequencies
    pbar = tqdm(total=vocab_size - len(vocab), desc="BPE merges")
    while len(vocab) < vocab_size:
        # pick the most frequent pair from pair_freqs
        pair_to_merge = get_pair_to_merge(pair_freqs)
        # print(f"[DBG] pair to merge: {pair_to_merge}")
        new_vocab = pair_to_merge[0] + pair_to_merge[1]

        # add the new merged token to vocab, increment next_id
        vocab[next_id] = new_vocab
        next_id += 1
        # append the pair to merges
        merges.append(pair_to_merge)

        affected_pretokens = pair_positions.pop(pair_to_merge, set())
        del pair_freqs[pair_to_merge]

        for pretoken_id in affected_pretokens:
            freq = pretoken_freqs[pretoken_id]
            old_subwords = pretoken_subwords[pretoken_id]

            # remove all old pairs (pair_to_merge already removed globally)
            for pair in zip(old_subwords[:-1], old_subwords[1:]):
                if pair != pair_to_merge:
                    pair_freqs[pair] -= freq
                    pair_positions[pair].discard(pretoken_id)

            # build new subwords by merging all occurrences of pair_to_merge
            new_subwords = []
            i = 0
            while i < len(old_subwords):
                if i < len(old_subwords) - 1 and (old_subwords[i], old_subwords[i+1]) == pair_to_merge:
                    new_subwords.append(new_vocab)
                    i += 2
                else:
                    new_subwords.append(old_subwords[i])
                    i += 1

            pretoken_subwords[pretoken_id] = new_subwords

            # add all new pairs
            for new_pair in zip(new_subwords[:-1], new_subwords[1:]):
                pair_positions[new_pair].add(pretoken_id)
                pair_freqs[new_pair] += freq
        pbar.update(1)
    # print(f"[DBG] final merges: {merges}")
    return vocab, merges


if __name__ == "__main__":
    import sys
    import pickle

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "train":
        if len(sys.argv) < 5:
            print("Usage: python -m src.train_bpe train <input_path> <output_pkl> <vocab_size> [special_tokens...]")
            sys.exit(1)
        _, _, input_path, output_path, vocab_size_str, *special_tokens = sys.argv
        if not special_tokens:
            special_tokens = ["<|endoftext|>"]
        print(f"Training BPE on {input_path}, vocab_size={vocab_size_str}, special_tokens={special_tokens}")
        vocab, merges = train_bpe(input_path, int(vocab_size_str), special_tokens)
        with open(output_path, "wb") as f:
            pickle.dump({"vocab": vocab, "merges": merges}, f)
        print(f"Saved to {output_path}")

    elif cmd == "read":
        if len(sys.argv) < 3:
            print("Usage: python -m src.train_bpe read <pkl_path>")
            sys.exit(1)
        pkl_path = sys.argv[2]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        vocab, merges = data["vocab"], data["merges"]
        print(f"Vocab size: {len(vocab)}")
        print(f"Merges: {len(merges)}\n")
        # print("First 20 vocab entries:")
        # for i in range(min(20, len(vocab))):
        #     print(f"  {i}: {vocab[i]}")
        # print("\nLast 20 vocab entries:")
        # for i in range(max(0, len(vocab) - 20), len(vocab)):
        #     print(f"  {i}: {vocab[i]}")
        # print("\nFirst 20 merges:")
        # for a, b in merges[:20]:
        #     print(f"  {a} + {b} -> {a + b}")

    else:
        print("Usage:")
        print("  python -m src.train_bpe train <input_path> <output_pkl> <vocab_size> [special_tokens...]")
        print("  python -m src.train_bpe read <pkl_path>")
        sys.exit(1)
