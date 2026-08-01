"""A small word-level tokenizer for radiology reports.

Deliberately simple and local: it runs on CPU in microseconds and has no
downloads, so the alignment machinery can be built and debugged before a real
medical LLM is introduced. The one non-obvious choice is number handling --
see `bucket_number`.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"
SPECIAL = (PAD, UNK, CLS)

_TOKEN_RE = re.compile(r"[a-z]+|\d+\.?\d*|%")

# Magnitude buckets in millimetres. Lesion sizes cluster below 100mm, so the
# bins are finer there and coarser above.
NUMBER_BUCKETS = ((0, 10), (10, 20), (20, 30), (30, 40),
                  (40, 60), (60, 90), (90, 150), (150, 10_000))


def bucket_number(token: str) -> str:
    """Map a numeric token to a magnitude bucket.

    Keeping raw numbers as vocabulary items means a 45mm lesion unseen in
    training is out-of-vocabulary at test time. Bucketing lets the model learn
    'a lesion of roughly this size' and generalise, while still distinguishing
    32mm from 38mm when they fall in different bins.
    """
    try:
        value = float(token)
    except ValueError:
        return token
    for lo, hi in NUMBER_BUCKETS:
        if lo <= value < hi:
            return f"<num{lo}-{hi}>"
    return "<num>"


def tokenize(text: str, bucket_numbers: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [bucket_number(t) for t in tokens] if bucket_numbers else tokens


class ReportTokenizer:
    """Vocabulary built from a corpus, with encode/decode."""

    def __init__(self, vocab: list[str], max_length: int = 192,
                 bucket_numbers: bool = True) -> None:
        self.itos = list(vocab)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.max_length = max_length
        self.bucket_numbers = bucket_numbers

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 1,
              max_length: int = 192, bucket_numbers: bool = True) -> "ReportTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text, bucket_numbers))
        kept = sorted(t for t, n in counts.items() if n >= min_count)
        return cls([*SPECIAL, *kept], max_length, bucket_numbers)

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        """-> (token_ids, attention_mask), padded/truncated to max_length."""
        unk = self.stoi[UNK]
        ids = [self.stoi[CLS]]
        ids += [self.stoi.get(t, unk) for t in tokenize(text, self.bucket_numbers)]
        ids = ids[:self.max_length]
        mask = [1] * len(ids)
        pad = self.max_length - len(ids)
        return ids + [self.pad_id] * pad, mask + [0] * pad

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.itos[i] for i in ids
                        if self.itos[i] not in (PAD, CLS))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "vocab": self.itos, "max_length": self.max_length,
            "bucket_numbers": self.bucket_numbers}), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ReportTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"], data["max_length"], data["bucket_numbers"])