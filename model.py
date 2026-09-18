from torch import nn
import torch
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, X):
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        batch, sequence, _ = X.shape

        Q = Q.reshape(
            batch,
            sequence,
            self.num_heads,
            self.d_head
        )
        K = K.reshape(
            batch,
            sequence,
            self.num_heads,
            self.d_head
        )
        V = V.reshape(
            batch,
            sequence,
            self.num_heads,
            self.d_head
        )

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scaled_scores = scores / math.sqrt(self.d_head)

        mask = torch.tril(
            torch.ones(
                sequence,
                sequence,
                device=X.device,
                dtype=torch.bool
            )
        )

        masked_scores = scaled_scores.masked_fill(
            ~mask,
            -math.inf
        )

        attention_weights = torch.softmax(
            masked_scores,
            dim=-1
        )

        attention_output = torch.matmul(
            attention_weights,
            V
        )

        attention_output = attention_output.transpose(1, 2)

        attention_output = attention_output.reshape(
            batch,
            sequence,
            self.d_model
        )

        attention_output = self.o_proj(attention_output)

        return attention_output


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(
            d_model,
            d_ff
        )

    def forward(self, x):
        x_norm = self.ln1(x)
        x = x + self.attention(x_norm)

        x_norm = self.ln2(x)
        x = x + self.ffn(x_norm)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_sequence_length,
        d_model,
        num_heads,
        d_ff,
        num_layers
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff
            )
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x):
        _, sequence = x.shape

        assert sequence <= self.max_sequence_length, \
            "Sequence length exceeds max_sequence_length"

        token_embeddings = self.token_embedding(x)

        positions = torch.arange(
            sequence,
            device=x.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)

        logits = self.lm_head(x)

        return logits