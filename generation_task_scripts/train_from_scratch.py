import json, math, re, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split

# ======= CONFIG =======
SEQ_LEN = 10
BATCH_SIZE = 128
EPOCHS = 10
MIN_FREQ = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= TOKENIZATION =======
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def simple_tokenize(text):
    return re.findall(r"\w+|\S", text.lower())

def build_vocab(tokens_list, min_freq):
    counter = Counter(tok for line in tokens_list for tok in line)
    vocab = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + sorted([t for t, c in counter.items() if c >= min_freq])
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return vocab, word2idx, idx2word

def numericalize(tokens, word2idx):
    return [word2idx.get(tok, word2idx["<UNK>"]) for tok in tokens]

# ======= DATA LOADING =======
def load_headlines(json_path):
    with open(json_path, 'r') as f:
        return [json.loads(l)["headline"] for l in f if json.loads(l)["is_sarcastic"] == 1]

def make_sequence_dataset(headlines, word2idx, seq_len):
    data = []
    for line in headlines:
        tokens = ["<BOS>"] + simple_tokenize(line) + ["<EOS>"]
        ids = numericalize(tokens, word2idx)
        if len(ids) < seq_len + 1:
            ids += [word2idx["<PAD>"]] * (seq_len + 1 - len(ids))
        else:
            ids = ids[:seq_len + 1]
        data.append(ids)
    return data

def get_sequence_batches(data, batch_size):
    np.random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch = torch.tensor(batch, dtype=torch.long)
        x = batch[:, :-1]
        y = batch[:, 1:]
        yield x.to(DEVICE), y.to(DEVICE)

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
    def __init__(self, d_model=768, nhead=4, num_layers=2, dim_feedforward=256):
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

# ======= TRAINING =======
def train(model, dataset, word2idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=2e-4):
    
    train_data, val_data = train_test_split(dataset, test_size=0.1)
    
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        batches = get_sequence_batches(train_data, batch_size)
        batch_num = 0
        for x, y in batches:
            # if batch_num >= 8000:
            #     break
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/batch_num:.4f}")
        
        model.eval()
        val_loss = 0
        batch_num = 0
        with torch.no_grad():
            for x, y in get_sequence_batches(val_data, batch_size):
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
                batch_num += 1
        print(f"Validation Loss: {val_loss/batch_num:.4f}")
        model.train()

# ======= GENERATION =======
def generate(model, word2idx, idx2word, max_len=10, temp=1.0):
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
    path = "Sarcasm_Headlines_Dataset.json"
    if not os.path.exists(path):
        print("Dataset not found!")
        return

    headlines = load_headlines(path)
    tokenized = [simple_tokenize(h) for h in headlines]
    vocab, word2idx, idx2word = build_vocab(tokenized, MIN_FREQ)
    print(f"Loaded {len(headlines)} headlines.")
    print(f"Vocab size: {len(vocab)}")

    dataset = make_sequence_dataset(headlines, word2idx, SEQ_LEN)
    
    model = TransformerLM()

    print("\nTraining on", DEVICE)
    train(model, dataset, word2idx)

    torch.save(model.state_dict(), "transformer_bert_lm_sarcastic.pth")
    import pickle
    with open("vocab_sarcastic.pkl", "wb") as f:
        pickle.dump((vocab, word2idx, idx2word), f)
        
    print("\nGenerated Samples:")
    for i in range(10):
        print(f"'{generate(model, word2idx, idx2word, temp=1.0)}'")

if __name__ == "__main__":
    main()
