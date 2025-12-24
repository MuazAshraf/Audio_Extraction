from __future__ import annotations

from typing import List


class CharTokenizer:
    """Simple character-level tokenizer with special tokens."""

    def __init__(self, texts: List[str]):
        chars = sorted(set("".join(t.lower() for t in texts)))
        specials = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.itos = specials + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.sos_id = self.stoi["<sos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    def encode(self, text: str) -> List[int]:
        text = text.lower()
        ids = [self.sos_id]
        ids += [self.stoi.get(c, self.unk_id) for c in text]
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        out = []
        for t in ids:
            if t == self.eos_id or t == self.pad_id:
                break
            if t == self.sos_id:
                continue
            if t < len(self.itos):
                out.append(self.itos[t])
        return "".join(out)

    def __len__(self) -> int:
        return len(self.itos)
