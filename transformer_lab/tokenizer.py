import json
import os
from collections.abc import Iterable, Iterator
from functools import lru_cache

import regex as re


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        if special_tokens is None or len(special_tokens) == 0:
            self.special_tokens = None
        else:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            vocab_set = set(self.vocab.values())
            for special_token in self.special_tokens:
                if special_token.encode("utf-8") not in vocab_set:
                    self.vocab[len(self.vocab)] = special_token.encode("utf-8")
            self.specials_pat = re.compile("|".join([re.escape(x) for x in self.special_tokens]))
        self.vocab_inverse = {value: key for key, value in self.vocab.items()}
        self.merges_inverse = {value: idx for idx, value in enumerate(self.merges)}
        self.encode_cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_raw = json.load(f)
            vocab = {
                vocab_index: bytes([gpt2_byte_decoder[token] for token in vocab_item])
                for vocab_item, vocab_index in vocab_raw.items()
            }

        with open(merges_filepath, encoding="utf-8") as f:
            merges_raw = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in merges_raw
            ]

        return cls(vocab, merges, special_tokens)

    def encode_pre_token(self, pre_token: str) -> list[int]:
        if pre_token in self.encode_cache:
            return self.encode_cache[pre_token]

        pre_token_bytes = list(bytes([x]) for x in pre_token.encode("utf-8"))

        while True:
            lowest_rank = (len(self.merges), -1)

            for idx in range(len(pre_token_bytes) - 1):
                if (pre_token_bytes[idx], pre_token_bytes[idx + 1]) in self.merges_inverse:
                    lowest_rank = min(
                        lowest_rank, (self.merges_inverse[(pre_token_bytes[idx], pre_token_bytes[idx + 1])], idx)
                    )

            if lowest_rank[1] == -1:
                break

            pre_token_bytes.pop(lowest_rank[1])
            pre_token_bytes[lowest_rank[1]] = self.merges[lowest_rank[0]][0] + self.merges[lowest_rank[0]][1]

        self.encode_cache[pre_token] = [self.vocab_inverse[x] for x in pre_token_bytes]

        return self.encode_cache[pre_token]

    def encode_chunk(self, text: str) -> list[int]:
        # like encode, but special tokens have been removed
        return [y for x in map(self.encode_pre_token, self.PAT.findall(text)) for y in x]

    def encode(self, text: str) -> list[int]:
        result = []
        if self.special_tokens is not None:
            matches = self.specials_pat.finditer(text)
            prev_match = 0
            for match in matches:
                chunk = text[prev_match : match.start()]
                result.extend(self.encode_chunk(chunk))
                prev_match = match.end()
                result.append(self.vocab_inverse[match.group().encode("utf-8")])

            last_chunk = text[prev_match:]
            result.extend(self.encode_chunk(last_chunk))
        else:
            result = self.encode_chunk(text)

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[x] for x in ids).decode("utf-8", errors="replace")
