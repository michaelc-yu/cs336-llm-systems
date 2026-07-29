from typing import Iterable, Iterator
import json
import regex as re
import collections


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)

def pretokenize_as_str(s: str, special_tokens: set[str]) -> list[str]:
    sorted_special = sorted(special_tokens, key=len, reverse=True)

    if not sorted_special:
        parts = [s]
    else:
        pattern = "|".join(f"({re.escape(t)})" for t in sorted_special)
        parts = re.split(pattern, s)

    res = []
    for part in parts:
        if part is None or part == "":
            continue
        if part in special_tokens:
            res.append(part)
        else:
            for m in COMPILED_PAT.finditer(part):
                res.append(m.group())
    return res

def pretokenize(
    text: str,
    special_tokens: list[str],
) -> list[list[bytes]]:
    special_set = set(special_tokens)
    res = []
    for pretoken in pretokenize_as_str(text, special_tokens):
        b = pretoken.encode("utf-8")
        if pretoken in special_set:
            res.append([b])
        else:
            res.append([b[i : i + 1] for i in range(len(b))])
    return res


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.bytes_to_id = {}
        for id, bt in vocab.items():
            self.bytes_to_id[bt] = id

    def encode(
        self,
        text: str,
    ) -> list[int]:
        pretokens = pretokenize(text, self.special_tokens)
        res = []
        for pretoken in pretokens:
            ids = self.encode_pretoken(pretoken)
            res.extend(ids)
        return res

    def encode_pretoken(
        self,
        pretoken: list[bytes],
    ) -> list[int]:
        subwords = list(pretoken)
        for left, right in self.merges:
            new_subwords = []
            i = 0

            while i < len(subwords):
                if i < len(subwords) - 1 and subwords[i] == left and subwords[i+1] == right:
                    new_subwords.append(left + right)
                    i += 2
                else:
                    new_subwords.append(subwords[i])
                    i += 1
            subwords = new_subwords
        return [self.bytes_to_id[s] for s in subwords]

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(
        self,
        ids: list[int],
    ) -> str:
        bytes_list = [self.vocab[id] for id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        pass
