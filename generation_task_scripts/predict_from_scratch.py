import json, math, re, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from transformers import BertModel, BertTokenizer

# ======= CONFIG =======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= TOKENIZATION =======
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ======= MODEL =======
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TransformerLM(nn.Module):
    def __init__(self, d_model=768, nhead=4, num_layers=4, dim_feedforward=512):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.embedding = self.bert.embeddings  # reuse embedding layer
        self.pos = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.2,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, self.bert.config.vocab_size)

    def generate_causal_mask(self, size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, input_ids):
        x = self.embedding(input_ids=input_ids)
        x = self.pos(x)
        seq_len = x.size(1)
        mask = self.generate_causal_mask(seq_len, x.device)
        x = self.encoder(x, mask=mask)
        return self.fc(x)

# ======= GENERATION =======
def generate(model, word2idx, idx2word, max_len=20, temp=1.0):
    model.eval()
    input_ids = [word2idx["<BOS>"]]
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits = model(x)[0, -1] / temp
            probs = F.softmax(logits, dim=0).cpu().numpy()
            next_id = np.random.choice(len(probs), p=probs)
            next_token = idx2word.get(next_id, "<UNK>")
            if next_token in ["<EOS>", "<PAD>", "<BOS>", "<UNK>"]:
                break
            input_ids.append(next_id)
    return " ".join(idx2word[i] for i in input_ids[1:])

# ======= MAIN =======
def main():
    import pickle
    with open("vocab_sarcastic.pkl", "rb") as f:
        vocab, word2idx, idx2word = pickle.load(f)

    # Reconstruct the model and load weights
    model = TransformerLM()
    model.load_state_dict(torch.load("transformer_bert_lm_sarcastic.pth", map_location=DEVICE))
    model.to(DEVICE)
    print("Model loaded to {}".format(DEVICE))
    model.eval()

    print("\nGenerated Samples:")
    for i in range(10):
        print(f"'{generate(model, word2idx, idx2word, temp=0.7)}'")

if __name__ == "__main__":
    main()
