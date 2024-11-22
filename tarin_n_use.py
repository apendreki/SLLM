import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)

        x = torch.matmul(attention, V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(x)

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn = self.attention(x, x, x, mask)
        x = self.norm1(x + attn)
        ff = self.ff(x)
        x = self.norm2(x + ff)
        return x

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)

# Custom Dataset
class MathDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_len):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        input_ids = self.tokenizer(question, self.max_len)
        target_ids = self.tokenizer(answer, self.max_len)
        return torch.tensor(input_ids), torch.tensor(target_ids)

# Tokenizer
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.pad_token = "<pad>"
        self.pad_idx = self.word2idx[self.pad_token]

    def __call__(self, sentence, max_len):
        tokens = sentence.split()
        token_ids = [self.word2idx.get(token, self.word2idx["<pad>"]) for token in tokens]
        return token_ids + [self.pad_idx] * (max_len - len(token_ids))

# Load Dataset from Kaggle
def load_math_dataset(data_dir):
    train_path = os.path.join(data_dir, "train-easy.csv")
    data = pd.read_csv(train_path)
    questions = data['Question'].tolist()
    answers = data['Answer'].tolist()
    return questions, answers

# Training Loop
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Main Script
if __name__ == "__main__":
    # Parameters
    vocab = ["<pad>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "x", "=", "?"]
    vocab_size = len(vocab)
    d_model = 128
    num_layers = 2
    num_heads = 4
    d_ff = 512
    max_len = 20
    dropout = 0.1
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4

    # Data Preparation
    data_dir = "./math-dataset"  # Change to your dataset directory
    questions, answers = load_math_dataset(data_dir)
    tokenizer = SimpleTokenizer(vocab)
    dataset = MathDataset(questions, answers, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Optimizer, Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    # Training
    for epoch in range(num_epochs):
        loss = train_model(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), "math_problem_generator.pth")
    print("Model saved successfully!")
