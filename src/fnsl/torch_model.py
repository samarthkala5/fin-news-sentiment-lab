"""Fine-tune DistilBERT for 3-class sentiment with a hand-written PyTorch training loop.

DistilBERT is a smaller BERT (6 Transformer layers instead of 12, trained to imitate
BERT): about 40% smaller and 60% faster, while keeping most of BERT's accuracy.
Fine-tuning starts from its pre-trained weights, adds a small, randomly initialised
classification head (two linear layers) on the output for the [CLS] token, and trains ALL
weights for a few epochs with a small learning rate. The loop is written out by hand
instead of using the Hugging Face Trainer, so each step (forward pass, loss, backward
pass, gradient clipping, optimizer step, learning-rate schedule) is visible and can be
explained.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from fnsl import config

MAX_GRAD_NORM = 1.0  # gradients longer than this (L2 norm over all weights) are scaled down


def get_device() -> torch.device:
    """The GPU if PyTorch can see one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(use_amp: bool) -> contextlib.AbstractContextManager:
    """Mixed precision on GPU: run the forward pass in float16 where that is safe.

    float16 halves activation memory and runs much faster on GPU tensor cores; PyTorch
    keeps numerically sensitive operations (softmax, losses) in float32. On CPU this
    returns a context that does nothing, so the same loop runs in plain float32.
    """
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def load_model(name_or_path: str | Path) -> torch.nn.Module:
    """Pre-trained DistilBERT (Hub id) or a saved checkpoint folder, with a 3-way head.

    For the Hub model, transformers prints a LOAD REPORT. The head's weights (pre_classifier,
    classifier) are MISSING: expected, the head is new and gets trained by us. The vocab_*
    weights are UNEXPECTED: they belong to the masked-word pre-training head, which a
    classifier does not use.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        name_or_path,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        # transformers 5 defaults to dtype="auto" (whatever dtype the checkpoint was saved
        # in). Keep float32 weights: autocast only makes temporary float16 copies for the
        # fast operations, and GradScaler cannot unscale float16 gradients.
        dtype=torch.float32,
    )


class TextDataset(Dataset):
    """Tokenised texts and their labels; item i is one example as a dict of plain lists.

    No padding happens here. The DataLoader's collate function pads each batch only up
    to its longest text ("dynamic padding"), which saves compute on short tweets.
    """

    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        if len(encodings["input_ids"]) != len(labels):
            raise ValueError(f"{len(encodings['input_ids'])} encodings but {len(labels)} labels")
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, list[int] | int]:
        item: dict[str, list[int] | int] = {key: ids[idx] for key, ids in self.encodings.items()}
        item["labels"] = self.labels[idx]  # the model computes the loss when it gets "labels"
        return item


def make_loader(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """Tokenise df["text"] once and wrap it in a DataLoader with dynamic padding.

    Texts longer than max_length tokens are cut off. Use shuffle=True for training and
    shuffle=False for evaluation, so predictions come out in the same order as df. The
    seeded generator makes the shuffle order the same on every run.
    """
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        max_length=max_length,
        # DistilBERT has no segment embeddings, so it does not use token_type_ids. The
        # transformers 5 tokenizer for this checkpoint still returns them by default.
        return_token_type_ids=False,
    )
    dataset = TextDataset(
        {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]},
        df["label"].astype(int).tolist(),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # Pads input_ids with the [PAD] id and attention_mask with 0 (= "ignore this
        # position"), then stacks the batch into tensors.
        collate_fn=DataCollatorWithPadding(tokenizer),
        generator=generator,
        num_workers=num_workers,
    )


def count_truncated(texts: list[str], tokenizer, max_length: int) -> int:
    """How many texts are longer than max_length tokens (incl. [CLS] and [SEP])."""
    lengths = [len(ids) for ids in tokenizer(texts)["input_ids"]]
    return sum(n > max_length for n in lengths)


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW with weight decay on the weight matrices only (the usual BERT recipe).

    Adam gives every weight its own step size from running averages of its gradients.
    AdamW applies weight decay (shrinking weights slightly towards zero every step, a mild
    regulariser) directly to the weights instead of mixing it into the gradient, which is
    what makes decay behave properly with Adam. Biases and LayerNorm scales are 1-D
    tensors: there are few of them, they hardly overfit, and pulling a LayerNorm scale
    towards zero would just shrink that layer's output, so they get no decay.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def make_scheduler(
    optimizer: torch.optim.Optimizer, num_training_steps: int, warmup_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then linear decay of the learning rate to 0.

    The rate climbs from 0 to its peak over the first `warmup_ratio` of all steps, then
    falls in a straight line to 0 at the last step. Warmup avoids large, noisy updates at
    the start, while the new classification head is still random and Adam's running
    averages are based on very few batches; the decay lets training settle at the end.
    """
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """One pass over the training data; returns the mean training loss per example.

    `scaler` is a torch.amp.GradScaler when training with mixed precision on a GPU,
    otherwise None (plain float32).
    """
    model.train()  # training mode: dropout is on (randomly zeroes activations)
    total_loss, n_examples = 0.0, 0
    for batch in loader:
        # input_ids, attention_mask and labels go to the same device as the model.
        batch = {key: value.to(device) for key, value in batch.items()}

        # 1) Clear old gradients: PyTorch ADDS each backward pass's gradients to the
        #    stored ones, so without this every update would mix in all earlier batches.
        optimizer.zero_grad()

        # 2) Forward pass. Because the batch contains "labels", the model also returns
        #    the cross-entropy loss between its 3 class scores (logits) and the labels.
        with autocast_context(scaler is not None):
            loss = model(**batch).loss

        if scaler is None:
            # 3) Backward pass: d(loss)/d(weight) for every weight, by backpropagation.
            loss.backward()
            # 4) Clip: if the gradient vector is longer than MAX_GRAD_NORM, scale it down,
            #    so one odd batch cannot make a huge step and wreck the pre-trained weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            # 5) Update every weight with AdamW, using the current learning rate.
            optimizer.step()
        else:
            # Same steps with mixed precision. Small float16 gradients can underflow to 0,
            # so the loss is multiplied by a large factor before backward(); unscale_()
            # divides the gradients back so clipping sees their true size.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            # Skips the update if any gradient overflowed to inf/NaN (then PyTorch may warn
            # that the scheduler stepped before the optimizer; that is harmless).
            scaler.step(optimizer)
            scaler.update()  # lower the factor after an overflow, slowly raise it otherwise

        # 6) The schedule is defined in optimizer steps (one per batch), so the learning
        #    rate moves after every batch, not once per epoch.
        scheduler.step()

        # Weight by batch size so a short last batch counts only as much as it should.
        total_loss += loss.item() * len(batch["labels"])
        n_examples += len(batch["labels"])
    return total_loss / max(n_examples, 1)


def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, float]:
    """Predicted label ids for every example in `loader` (in loader order) and the mean loss."""
    model.eval()  # evaluation mode: dropout off, so the same text always gets the same output
    predictions, total_loss, n_examples = [], 0.0, 0
    with torch.no_grad():  # no gradients are needed to predict: saves memory and time
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device.type == "cuda"):
                outputs = model(**batch)
            total_loss += outputs.loss.item() * len(batch["labels"])
            n_examples += len(batch["labels"])
            # The predicted class is the one with the highest score (logit).
            predictions.append(outputs.logits.argmax(dim=-1).cpu())
    return torch.cat(predictions).numpy(), total_loss / max(n_examples, 1)
