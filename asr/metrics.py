from __future__ import annotations

from typing import List


def _levenshtein(a, b) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def compute_cer(preds: List[str], refs: List[str]) -> float:
    total_edits, total_chars = 0, 0
    for p, r in zip(preds, refs):
        total_edits += _levenshtein(list(p), list(r))
        total_chars += max(1, len(r))
    return total_edits / total_chars


def compute_wer(preds: List[str], refs: List[str]) -> float:
    total_edits, total_words = 0, 0
    for p, r in zip(preds, refs):
        p_words, r_words = p.split(), r.split()
        total_edits += _levenshtein(p_words, r_words)
        total_words += max(1, len(r_words))
    return total_edits / total_words
