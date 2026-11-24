"""
Training Module

Contains the training and evaluation logic.

Responsibilities:
- Create batches from tokenized sentence sequences.
- Calculate training loss using Cross-Entropy.
- Evaluate model performance using validation loss.
- Update model parameters using the AdamW optimizer.
- Apply gradient clipping to stabilize training.

"""

import torch
import torch.nn.functional as F
import random


def get_batch(sequences, batch_size, device):
    """
    Select complete sentence sequences and pad them
    so they can be processed together in one batch.
    """
    selected = random.choices(sequences, k=batch_size)

    inputs = [seq[:-1] for seq in selected]
    targets = [seq[1:] for seq in selected]

    max_len = max(len(seq) for seq in inputs)

    x = torch.zeros((batch_size, max_len), dtype=torch.long)
    y = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, (input_seq, target_seq) in enumerate(zip(inputs, targets)):
        length = len(input_seq)
        x[i, :length] = input_seq
        y[i, :length] = target_seq

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_sequences, val_sequences,
                  batch_size, device, eval_iters=10):

    model.eval()

    results = {}

    for split, sequences in [
        ("train", train_sequences),
        ("val", val_sequences)
    ]:
        losses = []

        for _ in range(eval_iters):
            x, y = get_batch(sequences, batch_size, device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100
            )

            losses.append(loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()

    return results


def train_model(
    model,
    train_sequences,
    val_sequences,
    steps=150,
    batch_size=4,
    learning_rate=0.001,
    eval_interval=50,
    device="cpu"
):
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    for step in range(steps):

        model.train()

        x, y = get_batch(
            train_sequences,
            batch_size,
            device
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        if (step + 1) % eval_interval == 0:
            losses = estimate_loss(
                model,
                train_sequences,
                val_sequences,
                batch_size,
                device
            )

            print(
                f"Step {step + 1:4d} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Validation Loss: {losses['val']:.4f}"
            )

    return model
