import torch
from torch.utils.data import Dataset
import sys
from pathlib import Path

# --- Setup Imports ---
# Adjust path to import modules from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import constants and the Config class
try:
    from data import DATASET 
    from src.config import Config
except ImportError as e:
    print(f"Error importing dependencies: {e}.")
    raise

# --- Extract Hyperparameters ---
# The train/val split ratio is a data preparation constant, let's define it here
# A common choice is 90% for training
TRAIN_RATIO = 0.9 
BATCH_SIZE = Config.batch_size
BLOCK_SIZE = Config.block_size

# --- 1. Data Loading and Vocabulary Creation ---

# Read the entire dataset using the constant imported from data/__init__.py
with open(DATASET, 'r', encoding='utf-8') as f:
    text = f.read()

# Get all unique characters in the text
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

# Update the vocab_size in the Config class for consistency
Config.vocab_size = VOCAB_SIZE

# Create vocabulary: unique chars <-> indices (stoi) and indices <-> unique chars (itos)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# --- 2. Encode/Decode Helpers ---

def encode(s):
    """Converts a string to a list of integer tokens."""
    return [stoi[c] for c in s]

def decode(l):
    """Converts a list of integer tokens to a string."""
    return ''.join([itos[i] for i in l])

# --- 3. Train/Validation Split ---

# Encode the entire text into a single tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)

# Split into training and validation sets
n = int(TRAIN_RATIO * len(data))
train_data = data[:n]
val_data = data[n:]

# --- 4. CharDataset Class ---

class CharDataset(Dataset):
    """
    A PyTorch Dataset for character-level language modeling.
    Returns (x, y) pairs of token sequences based on BLOCK_SIZE.
    """
    def __init__(self, data):
        self.data = data
        self.block_size = BLOCK_SIZE

    def __len__(self):
        """Returns the number of possible (x, y) samples."""
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """Returns (input, target) sequence pair."""
        # x is the input sequence
        x = self.data[idx : idx + self.block_size]
        
        # y is the target sequence (x shifted by one token)
        y = self.data[idx + 1 : idx + self.block_size + 1]
        
        return x, y

# --- 5. get_batch Function ---

def get_batch(split):
    """
    Generates a random minibatch of input (xb) and target (yb) sequences 
    by sampling random starting indices.
    """
    data_source = train_data if split == 'train' else val_data
    
    # Calculate the maximum possible starting index for a block
    max_idx = len(data_source) - BLOCK_SIZE
    
    if max_idx <= 0:
        raise ValueError("Data source is too small for the given BLOCK_SIZE or is empty.")

    # Get BATCH_SIZE random starting indices
    ix = torch.randint(max_idx, (BATCH_SIZE,))

    # Stack the sequences to form the batches
    xb = torch.stack([data_source[i : i + BLOCK_SIZE] for i in ix])
    yb = torch.stack([data_source[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    
    return xb, yb

# --- Checkpoint Verification ---

if __name__ == '__main__':
    print(f"Data loading from: {DATASET}")
    print(f"Vocabulary size set to: {Config.vocab_size}")
    
    # Verify get_batch function (Checkpoint)
    if len(train_data) > BLOCK_SIZE:
        xb, yb = get_batch('train')
        print("-" * 30)
        print(f"get_batch Checkpoint (train split):")
        print(f"BATCH_SIZE (B): {BATCH_SIZE}, BLOCK_SIZE (T): {BLOCK_SIZE}")
        print(f"xb.shape (Input Batch): {xb.shape}") # ✅ Should be [B, T]
        print(f"yb.shape (Target Batch): {yb.shape}") # ✅ Should be [B, T]
    else:
        print("Error: Train data is too short to generate a batch.")