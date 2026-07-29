import collections
from collections import Counter

corpus = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""

# start with each byte as a token
# iteratively identifies most frequent adjacent pair of chars in corpus and merge them
# repeat process

vocab = {bytes([i]) for i in range(256)}
vocab.add("<|endoftext|>")

print(f"initial vocab: {vocab}\n")

words = corpus.split()
print(f"words: {words}\n")

frequencies = collections.defaultdict(int)

for word in words:
    chars = " ".join(list(word))
    frequencies[chars] += 1

print(f"Initial frequencies: {frequencies}\n")


def get_pair_freqs(frequencies):
    pairs = Counter()
    for word, freq in frequencies.items():
        chars = word.split()
        for i in range(len(chars)-1):
            pair = (chars[i], chars[i+1])
            pairs[pair] += freq
    return pairs


def merge(pair, freq, vocab, frequencies):
    merged_token = "".join(pair)
    print(f"merged token: {merged_token}\n")
    vocab.add(merged_token.encode('utf-8'))
    new_frequencies = {}

    for word, freq in frequencies.items():
        # print(word, freq)
        new_word = []
        lst = word.split()
        i = 0
        while i < len(lst):
            if i < len(lst)-1 and (lst[i], lst[i+1]) == pair:
                new_word.append(lst[i]+lst[i+1])
                i+=2
            else:
                new_word.append(lst[i])
                i+=1
        new_word = " ".join(new_word)
        # print(f"new word: {new_word}")
        new_frequencies[new_word] = freq
    return vocab, new_frequencies

def comp(pair):
    freq = pair_freqs.get(pair)
    return (freq, pair)

num_merges = 15
for i in range(num_merges):
    print(f"[DBG] Iteration {i+1}, {i} merges so far")
    pair_freqs = get_pair_freqs(frequencies)
    if not pair_freqs:
        break
    print(f"pair freqs: {pair_freqs}\n")
    largest_pair = max(pair_freqs, key=comp)
    freq = pair_freqs[largest_pair]
    print(f"largest pair: {largest_pair}\nfreq: {freq}\n")

    vocab, frequencies = merge(largest_pair, freq, vocab, frequencies)
    print(f"new vocab: {vocab}\n")
    print(f"frequencies: {frequencies}\n")
    # print(frequencies[largest_pair])
