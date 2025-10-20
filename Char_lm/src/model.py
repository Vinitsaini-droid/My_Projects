import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
from pathlib import Path

# Adjust path to import modules from the project root if running from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the Config class
try:
    from data import DATASET # Used only to satisfy potential relative path imports 
    from src.config import Config
except ImportError:
    print("Error: Could not import Config from src.config. Ensure data/__init__.py and src/config.py are correctly defined.")
    raise

# --- Constants derived from Config ---
N_EMBED = Config.n_embed
BLOCK_SIZE = Config.block_size
N_HEAD = Config.n_head
N_LAYER = Config.n_layer
DROPOUT = Config.dropout
VOCAB_SIZE = Config.vocab_size 
DEVICE = Config.device # ★ CORRECTED: Using Config.device

# Assert to ensure vocab_size has been set by dataset.py before model creation
assert VOCAB_SIZE is not None, "Config.vocab_size must be set after loading dataset (in dataset.py)."


# --- 1. Single Attention Head ---

class Head(nn.Module):
    """
    A single self-attention head with causal (masked) attention.
    """

    def __init__(self):
        super().__init__()
        
        # d_k (key dimension)
        head_size = N_EMBED // N_HEAD
        
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        
        # Causal mask buffer
        self.register_buffer('tril', torch.tril(
            torch.ones(BLOCK_SIZE, BLOCK_SIZE)
        ).to(DEVICE)) # ★ CORRECTED: Ensure tril is on the right device

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        # 1. Compute Q, K, V
        k = self.key(x)   
        q = self.query(x) 
        
        # 2. Compute attention scores (scaled dot-product)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) 
        
        # 3. Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # 4. Normalize weights with softmax
        wei = F.softmax(wei, dim=-1)
        self.dropout(wei)

        # 5. Aggregate values
        v = self.value(x) 
        out = wei @ v    
        
        return out


# --- 2. Multi-Head Attention ---

class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads run in parallel.
    """
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(N_HEAD)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# --- 3. Feed Forward Block ---

class FeedForward(nn.Module):
    """
    A simple two-layer MLP.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED), 
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


# --- 4. Transformer Block ---

class TransformerBlock(nn.Module):
    """
    Combines Multi-Head Attention + FeedForward + LayerNorm + Residuals.
    """
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        # 1. Multi-Head Attention with Residual connection
        x = x + self.sa(self.ln1(x))
        
        # 2. Feed Forward with Residual connection
        x = x + self.ffwd(self.ln2(x))
        
        return x


# --- 5. Transformer Language Model ---

class TransformerLanguageModel(nn.Module):
    """
    The full Decoder-only Transformer (GPT) Language Model.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Token and Positional Embeddings
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        
        # 2. Stack of Transformer Blocks
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYER)])
        
        # 3. Final Layer Normalization
        self.ln_f = nn.LayerNorm(N_EMBED)
        
        # 4. Final linear layer
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE)
        
        # Move the entire model to the specified device
        self.to(DEVICE) # ★ CORRECTED: Using the local DEVICE constant (which uses Config.device)

    def forward(self, idx, targets=None):
        """
        Calculates logits and optionally the loss.
        """
        B, T = idx.shape

        # 1. Compute token and positional embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, N_EMBED)
        
        # Create positional indices (0, 1, ..., T-1) on the correct device
        pos_indices = torch.arange(T, device=DEVICE) 
        pos_emb = self.position_embedding_table(pos_indices) # (T, N_EMBED)
        
        # Combine embeddings
        x = tok_emb + pos_emb # (B, T, N_EMBED)
        
        # 2. Pass through the stack of Transformer blocks
        x = self.blocks(x) 
        
        # 3. Final LayerNorm
        x = self.ln_f(x) 
        
        # 4. Project to vocabulary size to get logits
        logits = self.lm_head(x) # (B, T, VOCAB_SIZE)

        loss = None
        if targets is not None:
            # Reshape for cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # Calculate cross-entropy loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generates new tokens.
        """
        # idx is (B, T) array of indices
        for _ in range(max_new_tokens):
            
            # Crop idx to the last BLOCK_SIZE tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            
            # Get predictions (logits)
            logits, _ = self(idx_cond)
            
            # Focus on the last time step
            logits = logits[:, -1, :] 
            
            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            
            # Append sampled index
            idx = torch.cat((idx, idx_next), dim=1) 
            
        return idx


# --- Checkpoint Verification ---

if __name__ == '__main__':
    print("Running model.py checkpoint...")
    
    try:
        # Import the necessary functions and data for the test
        from src.dataset import get_batch, VOCAB_SIZE, encode, decode
    except ImportError:
        print("Error: Could not import necessary functions from src.dataset. Please ensure dataset.py is functional.")
        sys.exit(1)

    # 1. Update Config's vocab_size and ensure DEVICE is correct
    Config.vocab_size = VOCAB_SIZE
    print(f"Loaded VOCAB_SIZE: {Config.vocab_size}, Running on DEVICE: {Config.device}")

    # 2. Generate a dummy batch and move to device
    xb, yb = get_batch('train')
    xb = xb.to(Config.device)
    yb = yb.to(Config.device)
    
    # 3. Instantiate the model
    model = TransformerLanguageModel()

    # 4. Run the forward pass (Checkpoint)
    logits, loss = model(xb, yb)

    # Verification prints
    print("-" * 30)
    print("Forward Pass Checkpoint:")
    print(f"Logits shape: {logits.shape} (Expected: [{Config.batch_size}, {Config.block_size}, {Config.vocab_size}])") 
    print(f"Loss value: {loss.item():.4f}")

    # 5. Run the generate method
    print("-" * 30)
    print("Generation Checkpoint:")
    start_context = torch.tensor(encode("The "), dtype=torch.long, device=Config.device).unsqueeze(0) 
    generated_tokens = model.generate(start_context, max_new_tokens=10)
    
    generated_text = decode(generated_tokens[0].tolist())
    print(f"Generated text: '{generated_text}'")