import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in the paper.
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in the paper.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        """
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        return context, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (batch_size, query_len, d_model)
            key: (batch_size, key_len, d_model)
            value: (batch_size, value_len, d_model)
            mask: (batch_size, query_len, key_len)
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        value_len = value.size(1)
        
        # Linear transformations and reshape for multi-head
        Q = self.W_q(query).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, value_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        
        # Final linear transformation
        output = self.W_o(context)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Networks as described in the paper.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single encoder layer with self-attention and feed-forward.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Single decoder layer with masked self-attention, cross-attention, and feed-forward.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Masked self-attention with residual connection and layer norm
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with residual connection and layer norm
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """
    Full Transformer model as described in "Attention is All You Need".
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 d_ff: int = 2048, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Final linear layer
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Create padding mask for sequences."""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size: int) -> torch.Tensor:
        """Create look-ahead mask for decoder."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                src_padding_mask: Optional[torch.Tensor] = None,
                tgt_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            src_mask: Source mask
            tgt_mask: Target mask
            src_padding_mask: Source padding mask
            tgt_padding_mask: Target padding mask
        """
        
        # Embed and add positional encoding
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding (note: we need to transpose for positional encoding)
        src_embedded = src_embedded.transpose(0, 1)  # (src_len, batch_size, d_model)
        tgt_embedded = tgt_embedded.transpose(0, 1)  # (tgt_len, batch_size, d_model)
        
        src_embedded = self.positional_encoding(src_embedded)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Transpose back
        src_embedded = src_embedded.transpose(0, 1)  # (batch_size, src_len, d_model)
        tgt_embedded = tgt_embedded.transpose(0, 1)  # (batch_size, tgt_len, d_model)
        
        src_embedded = self.dropout(src_embedded)
        tgt_embedded = self.dropout(tgt_embedded)
        
        # Pass through encoder layers
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        # Pass through decoder layers
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
        
        # Final linear transformation
        output = self.final_linear(decoder_output)
        
        return output


def create_transformer_model(src_vocab_size: int, tgt_vocab_size: int, **kwargs) -> Transformer:
    """
    Factory function to create a Transformer model with default parameters.
    """
    default_params = {
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 2048,
        'max_len': 5000,
        'dropout': 0.1
    }
    
    # Update with any provided parameters
    default_params.update(kwargs)
    
    return Transformer(src_vocab_size, tgt_vocab_size, **default_params) 