""" Runs the complete pipeline """

import torch

from data.dataset import get_dataset
from decoder_only_transformer.tokenizer.tokenizer import Tokenizer
from decoder_only_transformer.model.decoder import Decoder
from decoder_only_transformer.training.train import train_model


# Load dataset
sentences = get_dataset()


# Split dataset
train_sentences = sentences[:16]
val_sentences = sentences[16:]


# Build tokenizer using training data only
tokenizer = Tokenizer(train_sentences)


# Encode each sentence separately
train_sequences = [
    torch.tensor(
        tokenizer.encode(sentence, add_eos=True),
        dtype=torch.long
    )
    for sentence in train_sentences
]

val_sequences = [
    torch.tensor(
        tokenizer.encode(sentence, add_eos=True),
        dtype=torch.long
    )
    for sentence in val_sentences
]


# Model configuration
embed_dim = 32
num_heads = 4
num_layers = 2
block_size = 8
dropout = 0.1

model = Decoder(
    vocab_size=tokenizer.vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size,
    dropout=dropout
)


# Train model
device = "cpu"

model = train_model(
    model=model,
    train_sequences=train_sequences,
    val_sequences=val_sequences,
    steps=150,
    batch_size=4,
    learning_rate=0.001,
    eval_interval=50,
    device=device
)


# Generate text
model.eval()

prompt = "I"
input_ids = tokenizer.encode(prompt)

if not input_ids:
    input_ids = [tokenizer.stoi[tokenizer.unk_token]]

context = torch.tensor(
    [input_ids],
    dtype=torch.long,
    device=device
)

with torch.no_grad():
    generated_ids = model.generate(
        context,
        max_new_tokens=10,
        temperature=1.0
    )

print("\nGenerated text:")
print(tokenizer.decode(generated_ids[0].tolist()))
