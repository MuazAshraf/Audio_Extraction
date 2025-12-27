from __future__ import annotations

from typing import List

import torch

from asr.tokenizer import CharTokenizer


@torch.no_grad()
def greedy_decode(model, mels: torch.Tensor, tokenizer: CharTokenizer, max_len: int = 200) -> torch.Tensor:
    """Autoregressive greedy decoding for Seq2SeqASR."""
    model.eval()
    mels = mels.to(next(model.parameters()).device)
    enc_out = model.encoder(mels)
    batch_size = mels.size(0)
    hidden = torch.zeros(1, batch_size, 512, device=enc_out.device)
    input_char = torch.full((batch_size,), tokenizer.sos_id, device=enc_out.device, dtype=torch.long)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=enc_out.device)
    preds = []

    for _ in range(max_len):
        context, _ = model.attention(hidden[-1], enc_out)
        prediction, hidden = model.decoder(input_char, hidden, context)
        next_char = prediction.argmax(1)
        preds.append(next_char)
        finished |= next_char.eq(tokenizer.eos_id)
        input_char = next_char
        if finished.all():
            break

    if len(preds) == 0:
        return torch.zeros(batch_size, 0, dtype=torch.long)
    return torch.stack(preds, dim=1).cpu()
