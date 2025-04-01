import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLLM:
    def __init__(self, vocab_size=10000, embedding_dim=256, num_layers=4, num_heads=8, max_seq_len=512):
        """
        Initialize a simple LLM
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Create a simple tokenizer (in real models, this would be much more sophisticated)
        self.token_to_id = {"[PAD]": 0, "[START]": 1, "[END]": 2}
        self.id_to_token = {0: "[PAD]", 1: "[START]", 2: "[END]"}
        
        # For demonstration, just add some simple tokens
        for i in range(3, vocab_size):
            token = f"token_{i}"
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Initialize model components
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Create transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
    
    def tokenize(self, text):
        """
        Very simplified tokenization (real tokenizers are much more complex)
        """
        # In a real system, this would use subword tokenization like BPE or WordPiece
        # This is just a simple placeholder that tokenizes by space
        tokens = ["[START]"] + text.split() + ["[END]"]
        token_ids = []
        
        for token in tokens:
            # Handle unknown tokens
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Use a random token ID for unknown tokens (for demonstration only)
                token_ids.append(np.random.randint(3, self.vocab_size))
        
        return token_ids
    
    def forward(self, input_ids):
        """
        Forward pass through the model
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            
        Returns:
            logits: Tensor of logits [batch_size, seq_len, vocab_size]
        """
        # Create positional IDs
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long)
        
        # Get embeddings
        token_embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        position_embeddings = self.pos_embedding(position_ids)  # [seq_len, embedding_dim]
        
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Pass through transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states)
        
        # Get output logits
        logits = self.output_layer(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def generate(self, input_text, max_length=50, temperature=1.0, top_k=50):
        """
        Generate text given an input prompt
        
        Args:
            input_text: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            
        Returns:
            generated_text: Generated text
        """
        # Tokenize input
        token_ids = self.tokenize(input_text)
        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        
        # Generate tokens auto-regressively
        for _ in range(max_length):
            # Forward pass
            with torch.no_grad():
                logits = self.forward(input_tensor)
                
            # Get the logits for the last token
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Convert to probabilities
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample from the distribution
            next_token_id = top_k_indices[torch.multinomial(top_k_probs, 1)]
            
            # Add the token to the sequence
            input_tensor = torch.cat([input_tensor, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Check if END token was generated
            if next_token_id.item() == self.token_to_id["[END]"]:
                break
        
        # Convert token IDs back to text
        generated_ids = input_tensor[0].tolist()
        generated_tokens = [self.id_to_token[idx] for idx in generated_ids]
        
        # Remove special tokens and join
        while "[START]" in generated_tokens:
            generated_tokens.remove("[START]")
        while "[END]" in generated_tokens:
            generated_tokens.remove("[END]")
            
        generated_text = " ".join(generated_tokens)
        return generated_text


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        """
        Simple transformer block
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
        """
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
            
        Returns:
            x: Output tensor [batch_size, seq_len, embedding_dim]
        """
        # Self-attention block with residual connection and layer norm
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)  # Residual connection and layer norm
        
        # Feed-forward block with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Residual connection and layer norm
        
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        """
        Multi-head attention module
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
        """
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
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
            
        Returns:
            output: Output tensor [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project back to embedding dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        output = self.output_proj(attn_output)
        
        return output


# Example of speculative decoding (simplified)
class SpeculativeDecoding:
    def __init__(self, target_model, draft_model):
        """
        Speculative decoding using a smaller draft model
        
        Args:
            target_model: The main LLM
            draft_model: A smaller, faster model for draft predictions
        """
        self.target_model = target_model
        self.draft_model = draft_model
    
    def generate(self, input_text, num_speculative_tokens=5, max_length=50):
        """
        Generate text using speculative decoding
        
        Args:
            input_text: Input text prompt
            num_speculative_tokens: Number of tokens to generate speculatively
            max_length: Maximum number of tokens to generate
            
        Returns:
            generated_text: Generated text
        """
        # Tokenize input
        token_ids = self.target_model.tokenize(input_text)
        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        
        generated_length = 0
        
        while generated_length < max_length:
            # Draft model generates several tokens speculatively
            draft_tensor = input_tensor.clone()
            draft_predictions = []
            
            for _ in range(num_speculative_tokens):
                with torch.no_grad():
                    draft_logits = self.draft_model.forward(draft_tensor)
                
                # Sample next token
                next_token_logits = draft_logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                
                # Add to draft predictions
                draft_predictions.append(next_token_id.item())
                draft_tensor = torch.cat([draft_tensor, next_token_id], dim=1)
                
                # Check if END token was generated
                if next_token_id.item() == self.target_model.token_to_id["[END]"]:
                    break
            
            # Target model verifies predictions
            extended_input = torch.cat([
                input_tensor, 
                torch.tensor([[draft_predictions]], dtype=torch.long)
            ], dim=1)
            
            with torch.no_grad():
                target_logits = self.target_model.forward(extended_input)
            
            # Find the first divergence (or accept all)
            accepted_tokens = []
            for i, draft_token in enumerate(draft_predictions):
                pos = input_tensor.shape[1] + i
                target_probs = F.softmax(target_logits[0, pos-1, :], dim=-1)
                
                # Sample from target distribution
                target_token = torch.multinomial(target_probs, 1).item()
                
                if target_token == draft_token:
                    accepted_tokens.append(draft_token)
                else:
                    # Divergence found, accept target token and stop
                    accepted_tokens.append(target_token)
                    break
            
            # Update input with accepted tokens
            for token in accepted_tokens:
                input_tensor = torch.cat([
                    input_tensor, 
                    torch.tensor([[token]], dtype=torch.long)
                ], dim=1)
            
            generated_length += len(accepted_tokens)
            
            # Check if generation should end
            if input_tensor[0, -1].item() == self.target_model.token_to_id["[END]"]:
                break
        
        # Convert token IDs back to text (same as before)
        generated_ids = input_tensor[0].tolist()
        generated_tokens = [self.target_model.id_to_token[idx] for idx in generated_ids]
        
        # Remove special tokens and join
        while "[START]" in generated_tokens:
            generated_tokens.remove("[START]")
        while "[END]" in generated_tokens:
            generated_tokens.remove("[END]")
            
        generated_text = " ".join(generated_tokens)
        return generated_text


# Example usage
def run_example():
    # Create a toy model
    model = SimpleLLM(vocab_size=10000, embedding_dim=256, num_layers=4, num_heads=8)
    
    # Create input
    input_text = "What is machine learning"
    
    print(f"Input: {input_text}")
    
    # Track token-by-token generation
    token_ids = model.tokenize(input_text)
    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    
    print("\nToken-by-token generation:")
    for i in range(10):  # Generate 10 tokens
        with torch.no_grad():
            logits = model.forward(input_tensor)
        
        # Get next token (simplified)
        next_token_id = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        token = model.id_to_token[next_token_id.item()]
        
        print(f"Token {i+1}: {token}")
        
        # Add token to input
        input_tensor = torch.cat([input_tensor, next_token_id], dim=1)
        
        # Check for end token
        if next_token_id.item() == model.token_to_id["[END]"]:
            break
    
    # Note: In a real implementation, the model weights would be trained,
    # and the output would be meaningful. This is just for demonstration.
    print("\nNote: This model has random weights and won't produce coherent text.")
    print("In a real LLM, the generated tokens would form a meaningful response.")


if __name__ == "__main__":
    run_example()