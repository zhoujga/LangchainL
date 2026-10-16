from __future__ import annotations

import argparse
import regex
from collections import Counter
from pathlib import Path


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 以 bytes 读取语料，之后显式按 UTF-8 解码，避免默认编码影响结果。
    with open(input_path, "rb") as f:
        corpus = f.read()
    text = corpus.decode("utf-8")

    # special token 是文档边界：不把它的内部字符交给预分词或 BPE 统计。
    # 这里先按照 special 把原始文本分开成 split list。
    if not special_tokens:
        split_list = [text]
    else:
        special_token_pattern = "|".join(
            regex.escape(special) for special in special_tokens
        )
        split_list = regex.split(special_token_pattern, text)

    # 再对拆开来的每一项，按照预分词器进行预分词。
    all_pretokens = []
    for split in split_list:
        all_pretokens.extend(regex.findall(PATTERN, split))

    # 先统计下预分词后每个词的频率，例如 "low" -> 5。
    pretoken_text_counts = Counter(all_pretokens)

    # 将每个 pre-token 从字符串转为 UTF-8 bytes，再拆成独立 byte token。
    # token_sequence_counts 的内容类似：(b"l", b"o", b"w") -> 5。
    token_sequence_counts = {}
    for pretoken, count in pretoken_text_counts.items():
        raw_bytes = pretoken.encode("utf-8")
        bytes_tokens = [bytes([byte_value]) for byte_value in raw_bytes]
        token_sequence_counts[tuple(bytes_tokens)] = count

    # 初始词表包含全部 256 个可能的单 byte 值。
    # special token 作为一个完整 token 加入 vocab，不拆成内部的 byte token。
    merges = []
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    for token_id, special_token in enumerate(special_tokens, start=256):
        vocab[token_id] = special_token.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size 至少需要为 {len(vocab)}，以容纳基础 byte 和 special token。"
        )

    # 每轮只学习一种最高频 pair；新 token 会让 vocab 的大小增加 1。
    current_vocab_size = len(vocab)
    while current_vocab_size < vocab_size:
        # 基于当前 token 序列统计相邻 pair 的加权次数。
        pair_counts = Counter()
        for sequence, count in token_sequence_counts.items():
            for pair in zip(sequence, sequence[1:]):
                pair_counts[pair] += count

        # 没有相邻 pair 时，说明无法继续学习新的 merge。
        if not pair_counts:
            break

        # 同频时由 find_max 按字典序选择更大的 pair。
        pair, _ = find_max(pair_counts)
        target = pair[0] + pair[1]
        vocab[current_vocab_size] = target

        # merges 保留创建顺序；编码时会按这个顺序应用规则。
        merges.append(pair)

        # 在新表里记录 merge 后的序列，避免边遍历边修改旧表。
        new_token_sequence_counts = {}
        for sequence, count in token_sequence_counts.items():
            new_sequence = []
            index = 0
            while index < len(sequence):
                if (
                    index + 1 < len(sequence)
                    and (sequence[index], sequence[index + 1]) == pair
                ):
                    # 命中时消费两个旧 token，写入一个新 token，因此不会重叠合并。
                    new_sequence.append(target)
                    index += 2
                else:
                    new_sequence.append(sequence[index])
                    index += 1
            new_token_sequence_counts[tuple(new_sequence)] = count

        token_sequence_counts = new_token_sequence_counts
        current_vocab_size += 1

    return vocab, merges


def find_max(pair_counts: Counter) -> tuple[tuple[bytes, bytes], int]:
    return max(pair_counts.items(), key=lambda item: (item[1], item[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-pair encoding tokenizer.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("knowledge.txt"),
        help="UTF-8 text corpus (default: knowledge.txt beside this script)",
    )
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=None,
        help="special token; may be supplied more than once",
    )
    args = parser.parse_args()
    special_tokens = args.special_tokens or ["<|endoftext|>"]

    vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens)
    print(f"corpus: {args.input}")
    print(f"vocab size: {len(vocab)}")
    print(f"merge count: {len(merges)}")
    print("first merges:")
    for pair in merges[:10]:
        print(f"  {pair[0]!r} + {pair[1]!r} -> {pair[0] + pair[1]!r}")


if __name__ == "__main__":
    main()
