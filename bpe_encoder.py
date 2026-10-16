from __future__ import annotations

import regex
from collections.abc import Iterable, Iterator
from pathlib import Path

from bpe_tokenizer import PATTERN, train_bpe


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        if special_tokens is None:
            special_tokens = []

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # Special tokens are atomic entries in the vocabulary.
        for special_token in special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in self.vocab.values():
                self.vocab[max(self.vocab.keys()) + 1] = special_token_bytes

        # Build the inverse mapping needed by encode().
        self.vocab_b2i = {value: key for key, value in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        token_ids = []

        if not self.special_tokens:
            split_list = [text]
        else:
            special_token_pattern = "|".join(
                regex.escape(special_token)
                for special_token in sorted(self.special_tokens, key=len, reverse=True)
            )
            split_list = regex.split(f"({special_token_pattern})", text)

        for part in split_list:
            if not part:
                continue

            if part in self.special_tokens:
                token_ids.append(self.vocab_b2i[part.encode("utf-8")])
                continue

            # Apply the same pre-tokenization used during BPE training.
            for pre_token in regex.findall(PATTERN, part):
                raw_bytes = pre_token.encode("utf-8")
                token_sequence = [bytes([byte_value]) for byte_value in raw_bytes]

                # Apply merges in training order, scanning each sequence left to right.
                for merge in self.merges:
                    merged_sequence = []
                    index = 0
                    while index < len(token_sequence) - 1:
                        if (token_sequence[index], token_sequence[index + 1]) == merge:
                            merged_sequence.append(token_sequence[index] + token_sequence[index + 1])
                            index += 2
                        else:
                            merged_sequence.append(token_sequence[index])
                            index += 1

                    if index == len(token_sequence) - 1:
                        merged_sequence.append(token_sequence[index])
                    token_sequence = merged_sequence

                token_ids.extend(self.vocab_b2i[token] for token in token_sequence)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for part in iterable:
            yield from self.encode(part)

    def decode(self, ids: list[int]) -> str:
        encoded_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return encoded_bytes.decode("utf-8", errors="replace")


def main() -> None:
    corpus_path = Path(__file__).with_name("knowledge.txt")
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        input_path=corpus_path,
        vocab_size=300,
        special_tokens=special_tokens,
    )
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    sample_text = "RAG 检索知识。<|endoftext|>你好，世界！"
    token_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(token_ids)

    print(f"vocab size: {len(tokenizer.vocab)}")
    print(f"merge count: {len(tokenizer.merges)}")
    print(f"token count: {len(token_ids)}")
    print(f"decoded: {decoded_text}")
    assert decoded_text == sample_text, "encode/decode round trip failed"

    iterable_ids = list(tokenizer.encode_iterable(["RAG ", "测试"]))
    assert tokenizer.decode(iterable_ids) == "RAG 测试"
    print("round-trip tests passed")


if __name__ == "__main__":
    main()
