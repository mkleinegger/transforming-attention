import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Embedding layer
class InputEmbeddings(nn.Module):
    def __init__(self, d_modal, vocab_size):
        super(InputEmbeddings, self).__init__()
        self.d_modal = d_modal
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_modal)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        return self.embeddings(x) * math.sqrt(self.d_modal)


# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(positions.float() * div_term)
        pe[:, 1::2] = torch.cos(positions.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        _, _, _, d_k = key.size()
        k_t = key.transpose(2, 3)
        score = (query @ k_t) / math.sqrt(d_k) 

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        score = self.softmax(score)
        value = score @ value
        return value
        
# Multi-head attention layer
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttentionBlock, self).__init__()
        self.embedding_dim = d_model
        self.self_attention = SelfAttention()
        self.num_heads = num_heads # The number of heads
        self.dim_per_head = d_model // num_heads # The dimension of each head
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size, _, _ = key.size()
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        output = self.self_attention(query, key, value, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        output = self.out(output)
        return output

# Feed forward layer
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.feed_forward = FeedForwardLayer(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        _x = x
        x = self.attention(query=x, key=x, value=x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        _x = x
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.attention1 = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)


        self.attention2 = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.feed_forward = FeedForwardLayer(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        _x = x
        x = self.attention1(query=x, key=x, value=x, mask=tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if encoder_output is not None:
            _x = x
            x = self.attention2(query=x, key=encoder_output, value=encoder_output, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.feed_forward(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x


# Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        N,
        d_ff,
        max_seq_length,
        dropout,
        device = "cpu",
        pad_idx=0,
    ):
        super().__init__()
        self.device = device
        self.pad_idx = pad_idx
        self.src_embedding = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embedding = InputEmbeddings(d_model, tgt_vocab_size)
        self.src_positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.tgt_positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)]
        )
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def generate_mask(self, src, tgt, pad_idx=0):
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)

        seq_length = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones(seq_length, seq_length)).type(torch.ByteTensor).to(self.device)
        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt, self.pad_idx)
        src = self.src_embedding(src)
        src = self.src_positional_encoding(src)
        
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_positional_encoding(tgt)

        for layer in self.encoder:
            src = layer(src, src_mask)
        
        for layer in self.decoder:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        return self.fc(tgt)


    def encode(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)        
        
        x = self.src_embedding(src)
        x = self.src_positional_encoding(x)
        
        for layer in self.encoder:
            x = layer(x, src_mask)
    
        return x, src_mask

    def decode(self, tgt, memory, src_mask):
        tgt_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(3)
        nopeak_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1))).type(torch.ByteTensor).to(self.device)
        tgt_mask = tgt_mask & nopeak_mask
        
        x = self.tgt_embedding(tgt)
        x = self.tgt_positional_encoding(x)
        
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
    
        return self.fc(x)