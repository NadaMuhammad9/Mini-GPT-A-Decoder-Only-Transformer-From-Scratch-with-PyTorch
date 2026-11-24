"""
Tokenizer Module

Implements a Tokenizer from scratch (Word-Level)

Responsibilities:
- Build the vocabulary from training sentences.
- Convert text into token IDs (Encoding).
- Convert token IDs back into text (Decoding).
- Handle unknown tokens using a special <unk> token.
- Append an <eos> token when required.
"""
import re

class Tokenizer:
    def __init__(self, sentences):
        self.unk_token = "<unk>"
        self.eos_token = "<eos>"

        all_tokens = []

        for sentence in sentences:
            all_tokens.extend(self.tokenize(sentence))

        self.vocab = sorted(set(all_tokens))
        self.vocab = [self.unk_token, self.eos_token] + self.vocab

        self.stoi = {
            token: i for i, token in enumerate(self.vocab)
        }

        self.itos = {
            i: token for token, i in self.stoi.items()
        }

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text)

    def encode(self, text, add_eos=False):
        tokens = self.tokenize(text)

        ids = [
            self.stoi.get(token, self.stoi[self.unk_token])
            for token in tokens
        ]

        if add_eos:
            ids.append(self.stoi[self.eos_token])

        return ids

    def decode(self, ids):
        tokens = []

        for idx in ids:
            token = self.itos[int(idx)]

            if token in (self.unk_token, self.eos_token):
                continue

            tokens.append(token)

        text = ""

        for token in tokens:
            if token in [".", ",", "!", "?", ";", ":"]:
                text = text.rstrip() + token
            else:
                if text:
                    text += " "
                text += token

        return text

    @property
    def vocab_size(self):
        return len(self.vocab)
