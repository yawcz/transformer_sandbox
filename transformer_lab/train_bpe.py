import os
from heapq import heapify_max, heappop_max, heappush_max
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def build_freq_dict(corpus: str):
    freq: dict[tuple[bytes, ...], int] = {}

    for word in re.findall(PAT, corpus):
        key = tuple(bytes([x]) for x in word.encode("utf-8"))
        freq[key] = freq.get(key, 0) + 1

    return freq


def do_merges(
    freq: dict[tuple[bytes, ...], int], vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    freq_copy = freq.copy()

    vocab: dict[int, bytes] = {}
    for idx, special_token in enumerate(special_tokens):
        vocab[idx] = special_token.encode("utf-8")
    for i in range(256):
        vocab[i + len(special_tokens)] = bytes([i])

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - 256 - len(special_tokens)

    assert num_merges >= 0

    adj_counts: dict[tuple[bytes, bytes], int] = {}
    adj_keys: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for key, value in freq_copy.items():
        for i1, i2 in zip(key, key[1:]):
            pair = (i1, i2)
            adj_counts[pair] = adj_counts.get(pair, 0) + value
            if pair not in adj_keys:
                adj_keys[pair] = set()
            adj_keys[pair].add(key)

    heap = []
    for key, value in adj_counts.items():
        heap.append((value, key))

    heapify_max(heap)

    for _ in range(num_merges):
        # find pair to merge
        if len(heap) == 0:
            break

        pair_freq, merge_pair = heappop_max(heap)

        while len(heap) > 0 and adj_counts[merge_pair] != pair_freq:
            pair_freq, merge_pair = heappop_max(heap)

        if adj_counts[merge_pair] != pair_freq:
            break

        assert adj_counts[merge_pair] >= 0

        if adj_counts[merge_pair] == 0:
            break

        to_remove = []

        # apply merge and update adj_counts
        for key in adj_keys[merge_pair]:
            value = freq_copy.pop(key)

            to_remove.append(key)

            new_key = []
            i = 0
            while i < len(key):
                if key[i : i + 2] == merge_pair:
                    # merge happens here
                    if len(new_key) > 0:
                        adj_counts[(new_key[-1], key[i])] -= value
                        heappush_max(heap, (adj_counts[(new_key[-1], key[i])], (new_key[-1], key[i])))
                    for j in [i, i + 1]:
                        if 0 <= j and j + 1 < len(key):
                            adj_counts[(key[j], key[j + 1])] -= value
                            heappush_max(heap, (adj_counts[(key[j], key[j + 1])], (key[j], key[j + 1])))
                    if len(new_key) > 0:
                        adj_counts[(new_key[-1], key[i] + key[i + 1])] = (
                            adj_counts.get((new_key[-1], key[i] + key[i + 1]), 0) + value
                        )
                        heappush_max(
                            heap, (adj_counts[(new_key[-1], key[i] + key[i + 1])], (new_key[-1], key[i] + key[i + 1]))
                        )
                    if i + 2 < len(key):
                        adj_counts[(key[i] + key[i + 1], key[i + 2])] = (
                            adj_counts.get((key[i] + key[i + 1], key[i + 2]), 0) + value
                        )
                        heappush_max(
                            heap, (adj_counts[(key[i] + key[i + 1], key[i + 2])], (key[i] + key[i + 1], key[i + 2]))
                        )
                    new_key.append(key[i] + key[i + 1])
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1

            new_key = tuple(new_key)

            for i1, i2 in zip(new_key, new_key[1:]):
                if (i1, i2) not in adj_keys:
                    adj_keys[(i1, i2)] = set()
                adj_keys[(i1, i2)].add(new_key)

            freq_copy[new_key] = freq_copy.get(new_key, 0) + value

        for key in to_remove:
            for i1, i2 in zip(key, key[1:]):
                adj_keys[(i1, i2)].discard(key)

        vocab[len(vocab)] = merge_pair[0] + merge_pair[1]
        merges.append(merge_pair)

    return vocab, merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found = False

            # Find the special token in the mini chunk
            for special_token in split_special_tokens:
                found_at = mini_chunk.find(special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    found = True
                    break

            if found:
                break

            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def build_freq_dict_from_chunk(file_path: str, start: int, end: int, special_tokens) -> dict[tuple[bytes, ...], int]:
    chunk_freq: dict[tuple[bytes, ...], int] = {}

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        if len(special_tokens) > 0:
            split_chunk = re.split("|".join(re.escape(special_token) for special_token in special_tokens), chunk)
        else:
            split_chunk = [chunk]
        for subchunk in split_chunk:
            if not subchunk:
                continue
            subchunk_freq = build_freq_dict(subchunk)
            for key, value in subchunk_freq.items():
                chunk_freq[key] = chunk_freq.get(key, 0) + value

    return chunk_freq


def build_freq_dict_from_file(
    file_path: str | os.PathLike, special_tokens: list[str], num_processes=4
) -> dict[tuple[bytes, ...], int]:
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, [x.encode("utf-8") for x in special_tokens])

        freq: dict[tuple[bytes, ...], int] = {}

        with Pool(processes=num_processes) as pool:
            chunks_freq = pool.starmap(
                build_freq_dict_from_chunk,
                zip([file_path] * len(boundaries), boundaries, boundaries[1:], [special_tokens] * len(boundaries)),
            )

            for chunk_freq in chunks_freq:
                for key, value in chunk_freq.items():
                    freq[key] = freq.get(key, 0) + value

    return freq
