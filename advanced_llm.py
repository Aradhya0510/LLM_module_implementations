import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Position Embedding (RoPE) as described in 
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Generate θᵢ values
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_heads, head_dim]
            seq_len: Sequence length (optional, calculated from x if not provided)
            
        Returns:
            Tensor with rotary position embeddings applied
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # Create position indices
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [seq_len, 1]
        
        # Compute sinusoids
        sincos_inputs = position * self.inv_freq  # [seq_len, dim//2]
        sin = torch.sin(sincos_inputs)  # [seq_len, dim//2]
        cos = torch.cos(sincos_inputs)  # [seq_len, dim//2]
        
        # Apply rotation
        # Reshape x to [batch_size, seq_len, num_heads, head_dim//2, 2]
        dim = x.shape[-1]
        x1 = x[..., :dim//2]
        x2 = x[..., dim//2:dim]
        
        # Rotate using complex multiplication
        # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
        
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Concatenate the rotated components
        rotated_x = torch.cat([rotated_x1, rotated_x2], dim=-1)
        
        return rotated_x


class AlibiPositionalBias(nn.Module):
    """
    Implements ALiBi (Attention with Linear Biases) as described in
    "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
    """
    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Create slopes for each attention head
        self.slopes = torch.Tensor([-(2**(-(8 - i))) for i in range(num_heads)])
        self.register_buffer("alibi_bias_slopes", self.slopes.view(1, num_heads, 1, 1))
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute ALiBi positional bias
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Bias tensor of shape [1, num_heads, seq_len, seq_len]
        """
        # Create position difference matrix
        pos = torch.arange(seq_len, device=self.alibi_bias_slopes.device)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)  # [seq_len, seq_len]
        
        # Make sure values are always negative by taking absolute and negating
        rel_pos = -torch.abs(rel_pos).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply different slopes for each head
        alibi_bias = rel_pos * self.alibi_bias_slopes  # [1, num_heads, seq_len, seq_len]
        
        return alibi_bias


class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention mechanisms"""
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.head_dim = embedding_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
    
    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _reshape_from_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor back from multi-head attention"""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)


class StandardMultiHeadAttention(MultiHeadAttentionBase):
    """Standard multi-head attention with full quadratic complexity"""
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, and values
        q = self._reshape_for_attention(self.q_proj(x))  # [B, H, S, D]
        k = self._reshape_for_attention(self.k_proj(x))  # [B, H, S, D]
        v = self._reshape_for_attention(self.v_proj(x))  # [B, H, S, D]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, S, S]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, S, D]
        
        # Reshape and project back
        output = self._reshape_from_attention(attn_output)
        output = self.output_proj(output)
        
        return output


class RoPEMultiHeadAttention(MultiHeadAttentionBase):
    """Multi-head attention with Rotary Position Embeddings (RoPE)"""
    def __init__(self, embedding_dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__(embedding_dim, num_heads)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, and values
        q = self._reshape_for_attention(self.q_proj(x))  # [B, H, S, D]
        k = self._reshape_for_attention(self.k_proj(x))  # [B, H, S, D]
        v = self._reshape_for_attention(self.v_proj(x))  # [B, H, S, D]
        
        # Apply RoPE to queries and keys
        q = self.rope(q)
        k = self.rope(k)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, S, S]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, S, D]
        
        # Reshape and project back
        output = self._reshape_from_attention(attn_output)
        output = self.output_proj(output)
        
        return output


class AlibiMultiHeadAttention(MultiHeadAttentionBase):
    """Multi-head attention with ALiBi (Attention with Linear Biases)"""
    def __init__(self, embedding_dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__(embedding_dim, num_heads)
        self.alibi = AlibiPositionalBias(num_heads, max_seq_len)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, and values
        q = self._reshape_for_attention(self.q_proj(x))  # [B, H, S, D]
        k = self._reshape_for_attention(self.k_proj(x))  # [B, H, S, D]
        v = self._reshape_for_attention(self.v_proj(x))  # [B, H, S, D]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, S, S]
        
        # Apply ALiBi bias
        alibi_bias = self.alibi(seq_len)
        scores = scores + alibi_bias
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, S, D]
        
        # Reshape and project back
        output = self._reshape_from_attention(attn_output)
        output = self.output_proj(output)
        
        return output


class FlashAttention(MultiHeadAttentionBase):
    """
    Simplified implementation of Flash Attention
    
    Note: This is a conceptual implementation only! The actual FlashAttention
    algorithm is implemented at the CUDA level for true efficiency gains.
    This implementation approximates the concept by computing attention in chunks.
    """
    def __init__(self, embedding_dim: int, num_heads: int, block_size: int = 64):
        super().__init__(embedding_dim, num_heads)
        self.block_size = block_size
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, and values
        q = self._reshape_for_attention(self.q_proj(x))  # [B, H, S, D]
        k = self._reshape_for_attention(self.k_proj(x))  # [B, H, S, D]
        v = self._reshape_for_attention(self.v_proj(x))  # [B, H, S, D]
        
        # Initialize output
        output = torch.zeros_like(q)
        
        # Compute attention in blocks for memory efficiency
        for i in range(0, seq_len, self.block_size):
            # Get current query block
            q_block = q[:, :, i:i+self.block_size]
            
            # Compute scaled attention scores for this block
            block_scores = torch.matmul(q_block, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # Apply mask if provided
            if mask is not None:
                block_mask = mask[:, :, i:i+self.block_size, :]
                block_scores = block_scores + block_mask
            
            # Apply softmax to get attention weights
            block_attn_weights = F.softmax(block_scores, dim=-1)
            
            # Apply attention weights to values
            block_output = torch.matmul(block_attn_weights, v)
            
            # Add to output
            output[:, :, i:i+self.block_size] = block_output
        
        # Reshape and project back
        output = self._reshape_from_attention(output)
        output = self.output_proj(output)
        
        return output


class SlidingWindowAttention(MultiHeadAttentionBase):
    """
    Sliding Window Attention that restricts each token to attend only to
    nearby tokens within a fixed window size
    """
    def __init__(self, embedding_dim: int, num_heads: int, window_size: int = 256):
        super().__init__(embedding_dim, num_heads)
        self.window_size = window_size
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, and values
        q = self._reshape_for_attention(self.q_proj(x))  # [B, H, S, D]
        k = self._reshape_for_attention(self.k_proj(x))  # [B, H, S, D]
        v = self._reshape_for_attention(self.v_proj(x))  # [B, H, S, D]
        
        # Compute scaled dot product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, S, S]
        
        # Create sliding window mask
        window_mask = torch.ones_like(scores) * float('-inf')
        for i in range(seq_len):
            window_start = max(0, i - self.window_size // 2)
            window_end = min(seq_len, i + self.window_size // 2 + 1)
            window_mask[:, :, i, window_start:window_end] = 0
        
        # Apply sliding window mask
        scores = scores + window_mask
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, S, D]
        
        # Reshape and project back
        output = self._reshape_from_attention(attn_output)
        output = self.output_proj(output)
        
        return output


class DilatedAttention(MultiHeadAttentionBase):
    """
    Dilated Attention where each token attends to tokens at increasing intervals
    Similar to dilated convolutions but for attention
    """
    def __init__(self, embedding_dim: int, num_heads: int, num_dilation_rates: int = 4):
        super().__init__(embedding_dim, num_heads)
        # Each head uses a different dilation rate for sparse attention
        self.dilation_rates = [2 ** i for i in range(num_dilation_rates)]
        # Cycle through rates if we have more heads than rates
        self.head_dilation_rates = [self.dilation_rates[i % len(self.dilation_rates)] 
                                   for i in range(num_heads)]
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, and values
        q = self._reshape_for_attention(self.q_proj(x))  # [B, H, S, D]
        k = self._reshape_for_attention(self.k_proj(x))  # [B, H, S, D]
        v = self._reshape_for_attention(self.v_proj(x))  # [B, H, S, D]
        
        # Initialize output
        attn_output = torch.zeros_like(q)
        
        # Process each head with its specific dilation rate
        for h in range(self.num_heads):
            dilation = self.head_dilation_rates[h]
            
            # Create dilated attention mask for this head
            dilated_mask = torch.ones(seq_len, seq_len, device=x.device) * float('-inf')
            
            # For each position, allow attention to positions at intervals of the dilation rate
            for i in range(seq_len):
                for j in range(seq_len):
                    if abs(i - j) % dilation == 0 or i == j:
                        dilated_mask[i, j] = 0
            
            # Compute scores for this head
            head_scores = torch.matmul(q[:, h], k[:, h].transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # Apply dilated mask
            head_scores = head_scores + dilated_mask.unsqueeze(0)
            
            # Apply additional mask if provided
            if mask is not None:
                head_scores = head_scores + mask[:, h]
            
            # Apply softmax to get attention weights
            head_attn_weights = F.softmax(head_scores, dim=-1)
            
            # Apply attention weights to values
            head_output = torch.matmul(head_attn_weights, v[:, h])
            
            # Add to output
            attn_output[:, h] = head_output
        
        # Reshape and project back
        output = self._reshape_from_attention(attn_output)
        output = self.output_proj(output)
        
        return output


class AttentionWithSink(MultiHeadAttentionBase):
    """
    Attention with sink tokens that absorb attention from context tokens
    This is similar to the technique used in Landmark Attention and Attention Sinks
    """
    def __init__(self, embedding_dim: int, num_heads: int, num_sink_tokens: int = 8):
        super().__init__(embedding_dim, num_heads)
        self.num_sink_tokens = num_sink_tokens
        
        # Learnable sink token embeddings
        self.sink_tokens = nn.Parameter(torch.randn(1, num_sink_tokens, embedding_dim))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Prepend sink tokens to the input sequence
        sink_tokens = self.sink_tokens.expand(batch_size, -1, -1)
        x_with_sinks = torch.cat([sink_tokens, x], dim=1)
        
        # Project to queries, keys, and values
        q = self._reshape_for_attention(self.q_proj(x_with_sinks))  # [B, H, S+sinks, D]
        k = self._reshape_for_attention(self.k_proj(x_with_sinks))  # [B, H, S+sinks, D]
        v = self._reshape_for_attention(self.v_proj(x_with_sinks))  # [B, H, S+sinks, D]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Adjust mask if provided
        if mask is not None:
            # Extend mask to include sink tokens
            sink_mask = torch.zeros(batch_size, self.num_heads, seq_len, self.num_sink_tokens, 
                                  device=mask.device)
            extended_mask = torch.cat([sink_mask, mask], dim=-1)
            scores = scores[:, :, self.num_sink_tokens:] + extended_mask
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, S, D]
        
        # Extract only the non-sink output
        attn_output = attn_output[:, :, self.num_sink_tokens:]
        
        # Reshape and project back
        output = self._reshape_from_attention(attn_output)
        output = self.output_proj(output)
        
        return output


class TwoStreamAttention(nn.Module):
    """
    Two-Stream Attention where a separate query stream attends over the main stream
    This is similar to what's used in models like Perceiver and some cross-attention setups
    """
    def __init__(self, embedding_dim: int, num_heads: int, latent_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.head_dim = embedding_dim // num_heads
        
        # Latent query stream
        self.latent_queries = nn.Parameter(torch.randn(1, latent_dim, embedding_dim))
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
    
    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _reshape_from_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor back from multi-head attention"""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Expand latent queries to batch size
        latent_q = self.latent_queries.expand(batch_size, -1, -1)
        
        # Project to queries from latent vector, keys and values from input
        q = self._reshape_for_attention(self.q_proj(latent_q))  # [B, H, L, D]
        k = self._reshape_for_attention(self.k_proj(x))         # [B, H, S, D]
        v = self._reshape_for_attention(self.v_proj(x))         # [B, H, S, D]
        
        # Compute cross-attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L, S]
        
        # Apply mask if provided
        if mask is not None:
            # Reshape mask for cross-attention
            cross_mask = mask.unsqueeze(2).expand(-1, -1, self.latent_dim, -1)
            scores = scores + cross_mask
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L, D]
        
        # Reshape and project back
        output = self._reshape_from_attention(attn_output)
        output = self.output_proj(output)
        
        return output


class TransformerWithModernAttention(nn.Module):
    """Transformer block that can use any of the modern attention mechanisms"""
    def __init__(self, embedding_dim: int, num_heads: int, attention_type: str = "standard", 
                 dropout: float = 0.1, **attention_kwargs):
        super().__init__()
        
        # Select attention mechanism
        if attention_type == "standard":
            self.attention = StandardMultiHeadAttention(embedding_dim, num_heads)
        elif attention_type == "rope":
            self.attention = RoPEMultiHeadAttention(embedding_dim, num_heads, **attention_kwargs)
        elif attention_type == "alibi":
            self.attention = AlibiMultiHeadAttention(embedding_dim, num_heads, **attention_kwargs)
        elif attention_type == "flash":
            self.attention = FlashAttention(embedding_dim, num_heads, **attention_kwargs)
        elif attention_type == "sliding_window":
            self.attention = SlidingWindowAttention(embedding_dim, num_heads, **attention_kwargs)
        elif attention_type == "dilated":
            self.attention = DilatedAttention(embedding_dim, num_heads, **attention_kwargs)
        elif attention_type == "attention_sink":
            self.attention = AttentionWithSink(embedding_dim, num_heads, **attention_kwargs)
        elif attention_type == "two_stream":
            self.attention = TwoStreamAttention(embedding_dim, num_heads, **attention_kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Layer normalization and feed-forward network
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LayerNorm architecture (as used in many modern Transformers)
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward block with residual connection
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x)
        x = x + ffn_output
        
        return x


class ModernLLM(nn.Module):
    """
    A simple LLM implementation that can use various modern attention mechanisms
    """
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 256, 
                 num_layers: int = 4, num_heads: int = 8, max_seq_len: int = 2048,
                 attention_type: str = "standard", dropout: float = 0.1, 
                 **attention_kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Create embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Position embeddings depend on attention type
        if attention_type == "rope" or attention_type == "alibi":
            # For RoPE and ALiBi, we don't need separate position embeddings
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Create transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerWithModernAttention(
                embedding_dim, num_heads, attention_type, dropout, **attention_kwargs
            ) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embedding_dim)
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Add position embeddings if the attention type requires it
        if self.pos_embedding is not None:
            if position_ids is None:
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            position_embeddings = self.pos_embedding(position_ids)
            embeddings = embeddings + position_embeddings
        
        # Create attention mask
        mask = None  # Could implement masking for causal LM here
        
        # Pass through transformer layers
        x = embeddings
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Get output logits
        logits = self.output_layer(x)
        
        return logits


def demonstrate_attention_mechanisms():
    """
    Function to demonstrate different attention mechanisms with tensor visualizations
    """
    # Create a small batch of random embeddings (batch_size=2, seq_len=10, embedding_dim=256)
    batch_size, seq_len, embedding_dim, num_heads = 2, 10, 256, 8
    x = torch.randn(batch_size, seq_len, embedding_dim)
    
    print(f"Input shape: {x.shape}")
    
    # Initialize different attention mechanisms
    standard_attn = StandardMultiHeadAttention(embedding_dim, num_heads)
    rope_attn = RoPEMultiHeadAttention(embedding_dim, num_heads)
    alibi_attn = AlibiMultiHeadAttention(embedding_dim, num_heads)
    flash_attn = FlashAttention(embedding_dim, num_heads)
    sliding_window_attn = SlidingWindowAttention(embedding_dim, num_heads, window_size=4)
    dilated_attn = DilatedAttention(embedding_dim, num_heads)
    sink_attn = AttentionWithSink(embedding_dim, num_heads, num_sink_tokens=2)
    two_stream_attn = TwoStreamAttention(embedding_dim, num_heads, latent_dim=4)
    
    # Run each attention mechanism and compare outputs
    std_output = standard_attn(x)
    rope_output = rope_attn(x)
    alibi_output = alibi_attn(x)
    flash_output = flash_attn(x)
    window_output = sliding_window_attn(x)
    dilated_output = dilated_attn(x)
    sink_output = sink_attn(x)
    two_stream_output = two_stream_attn(x)
    
    print(f"Standard attention output shape: {std_output.shape}")
    print(f"RoPE attention output shape: {rope_output.shape}")
    print(f"ALiBi attention output shape: {alibi_output.shape}")
    print(f"Flash attention output shape: {flash_output.shape}")
    print(f"Sliding window attention output shape: {window_output.shape}")
    print(f"Dilated attention output shape: {dilated_output.shape}")
    print(f"Attention with sink output shape: {sink_output.shape}")
    print(f"Two-stream attention output shape: {two_stream_output.shape}")
    
    # Compare attention pattern differences
    print("\nAttention mechanism comparison:")
    print(f"Standard vs RoPE difference: {torch.mean(torch.abs(std_output - rope_output)):.4f}")
    print(f"Standard vs ALiBi difference: {torch.mean(torch.abs(std_output - alibi_output)):.4f}")
    print(f"Standard vs Flash difference: {torch.mean(torch.abs(std_output - flash_output)):.4f}")
    print(f"Standard vs Window difference: {torch.mean(torch.abs(std_output - window_output)):.4f}")
    print(f"Standard vs Dilated difference: {torch.mean(torch.abs(std_output - dilated_output)):.4f}")
    print(f"Standard vs Sink difference: {torch.mean(torch.abs(std_output - sink_output)):.4f}")


if __name__ == "__main__":
    # Simple demo of different attention mechanisms
    demonstrate_attention_mechanisms()
    
    # Create a small LLM with RoPE attention
    model = ModernLLM(
        vocab_size=10000,
        embedding_dim=256,
        num_layers=2,
        num_heads=8,
        attention_type="rope",
        max_seq_len=1024
    )
    
    # Create sample input
    input_ids = torch.randint(0, 10000, (2, 16))
    
    # Get output logits
    output = model(input_ids)
    print(f"\nModel output shape: {output.shape}")  # Should be [2, 16, 10000]
    
    print("\nAvailable attention mechanisms:")
    attention_types = [
        "standard", "rope", "alibi", "flash", 
        "sliding_window", "dilated", "attention_sink", "two_stream"
    ]
    for attn_type in attention_types:
        print(f"- {attn_type}")
    
    print("\nDemonstration complete!")