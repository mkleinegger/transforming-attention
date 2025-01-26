import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Embedding layer
class InputEmbeddings(nn.Module):
    def __init__(self, d_modal, vocab_size):
        super(InputEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, d_modal)
        self.d_modal = d_modal

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_modal)


# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

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
        return x + self.pe[:, : x.size(1)]


# Multi-head attention block
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.heads = heads

        self.d_k = d_model // heads  # Dimension of vector seen by each head
        self.d_v = d_model // heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo

    @staticmethod
    def scale_dot_product_attention(d_k, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        return attention_output

    @staticmethod
    def split_head(x, heads, d_k):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, heads, d_k).transpose(1, 2)

    @staticmethod
    def combine_head(x, heads, d_k):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, heads * d_k)

    def forward(self, Q, K, V, mask):
        Q = MultiHeadAttentionBlock.split_head(self.w_q(Q), self.heads, self.d_k)
        K = MultiHeadAttentionBlock.split_head(self.w_k(K), self.heads, self.d_k)
        V = MultiHeadAttentionBlock.split_head(self.w_v(V), self.heads, self.d_v)

        attention_output = MultiHeadAttentionBlock.scale_dot_product_attention(
            self.d_k, Q, K, V, mask
        )
        attention_output = MultiHeadAttentionBlock.combine_head(
            attention_output, self.heads, self.d_k
        )
        return self.w_o(attention_output)


# Feed forward layer
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.attention = MultiHeadAttentionBlock(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardLayer(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(feed_forward_output))


# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.attention = MultiHeadAttentionBlock(d_model, num_heads)
        self.attention2 = MultiHeadAttentionBlock(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardLayer(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # First attention block
        attention_output = self.attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention_output))

        # Second attention block
        attention_output = self.attention2(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attention_output))

        # Feed forward layer
        feed_forward_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(feed_forward_output))


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
    ):
        super().__init__()
        self.src_embedding = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embedding = InputEmbeddings(d_model, tgt_vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)]
        )
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        seq_length = tgt.size(1)
        nopeak_mask = (
            1
            - torch.triu(
                torch.ones(1, seq_length, seq_length, device=src.device), diagonal=1
            )
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src = self.dropout_layer(self.positional_encoding(self.src_embedding(src)))
        tgt = self.dropout_layer(self.positional_encoding(self.tgt_embedding(tgt)))

        for layer in self.encoder:
            src = layer(src, src_mask)
        for layer in self.decoder:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        return self.fc(tgt)
