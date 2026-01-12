from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from asr.data import build_loaders, prepare_data
from asr.decode import greedy_decode
from asr.metrics import compute_cer, compute_wer
from asr.tokenizer import CharTokenizer
from cnn_encoder import Seq2SeqASR

# ---- Config ----------------------------------------------------------------
ARROW_SHARD = Path(
    "/Users/app/Desktop/SPRINGLab___libri_speech-100/default/0.0.0/"
    "c9c03efabb3ccf308018a18e20342158733d7e00"
)
MAX_SAMPLES = None  # None = use full dataset
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
BATCH_SIZE = 4
EPOCHS = 1
MEL_CFG = dict(sr=16000, n_fft=400, hop_length=160, n_mels=128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
PATIENCE = 2  # epochs without val WER improvement before early stop


# ---- Training / evaluation -------------------------------------------------
def train_one_epoch(model, loader, optimizer, pad_id: int) -> float:
    model.train()
    total_loss = 0.0
    for step, (mels, tokens) in enumerate(loader, 1):
        mels = mels.to(DEVICE)
        tokens = tokens.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(mels, tokens, teacher_forcing=0.5)
        target = tokens[:, 1 : outputs.size(1) + 1]
        loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            target.reshape(-1),
            ignore_index=pad_id,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if step % 10 == 0:
            avg = total_loss / step
            print(f"  train step {step}/{len(loader)} loss={avg:.4f}", flush=True)
    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_loss(model, loader, pad_id: int) -> float:
    model.eval()
    total_loss = 0.0
    for step, (mels, tokens) in enumerate(loader, 1):
        mels = mels.to(DEVICE)
        tokens = tokens.to(DEVICE)
        outputs = model(mels, tokens, teacher_forcing=0.0)
        target = tokens[:, 1 : outputs.size(1) + 1]
        loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            target.reshape(-1),
            ignore_index=pad_id,
        )
        total_loss += loss.item()
        if step % 10 == 0:
            avg = total_loss / step
            print(f"  eval step {step}/{len(loader)} loss={avg:.4f}", flush=True)
    return total_loss / max(1, len(loader))


@torch.no_grad()
def decode_and_score(model, loader, tokenizer: CharTokenizer, max_len: int = 200):
    pred_texts, ref_texts = [], []
    for mels, tokens in loader:
        pred_ids = greedy_decode(model, mels, tokenizer, max_len=max_len)
        tokens = tokens.cpu()
        for i in range(tokens.size(0)):
            pred_texts.append(tokenizer.decode(pred_ids[i].tolist()))
            ref_texts.append(tokenizer.decode(tokens[i].tolist()))
    cer = compute_cer(pred_texts, ref_texts)
    wer = compute_wer(pred_texts, ref_texts)
    return cer, wer, pred_texts, ref_texts


def main():
    print(f"Using device: {DEVICE}")
    splits, tokenizer = prepare_data(
        shard_path=ARROW_SHARD,
        max_samples=MAX_SAMPLES,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        mel_cfg=MEL_CFG,
    )
    print(
        f"Dataset sizes -> train: {len(splits.train)}, "
        f"val: {len(splits.val)}, test: {len(splits.test)}"
    )
    print(f"Vocab size: {len(tokenizer)}")

    train_loader, val_loader, test_loader = build_loaders(
        splits, tokenizer, batch_size=BATCH_SIZE, n_mels=MEL_CFG["n_mels"]
    )

    model = Seq2SeqASR(vocab_size=len(tokenizer)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_wer = float("inf")
    epochs_no_improve = 0
    ckpt_path = CHECKPOINT_DIR / "best_seq2seq_asr.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, tokenizer.pad_id)
        val_loss = eval_loss(model, val_loader, tokenizer.pad_id)
        val_cer, val_wer, _, _ = decode_and_score(model, val_loader, tokenizer, max_len=200)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_CER={val_cer:.3f} | val_WER={val_wer:.3f}"
        )

        if val_wer < best_wer:
            best_wer = val_wer
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({"model_state": best_state, "tokenizer_itos": tokenizer.itos}, ckpt_path)
            epochs_no_improve = 0
            print(f"  ✓ New best WER. Saved checkpoint to {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  No val WER improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= PATIENCE:
                print("  Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = eval_loss(model, test_loader, tokenizer.pad_id)
    cer, wer, pred_texts, ref_texts = decode_and_score(model, test_loader, tokenizer, max_len=200)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Greedy decode (test) -> CER: {cer:.3f} | WER: {wer:.3f}")
    for idx, (p, r) in enumerate(zip(pred_texts, ref_texts)):
        if idx >= 5:
            break
        print(f"[sample {idx}] pred: {p!r}")
        print(f"[sample {idx}] ref : {r!r}")


if __name__ == "__main__":
    main()
