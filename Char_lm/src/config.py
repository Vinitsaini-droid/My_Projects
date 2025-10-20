"""
config.py

Central store for all model, training, and data hyperparameters.
This allows easy experimentation and ensures parameters are consistent across 
training, generation, and utility scripts.
"""

import torch

# --- Device Configuration ---
# Automatically selects GPU if available, otherwise uses CPU.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Config:
    """
    Configuration class storing all hyperparameters for the mini-transformer.
    """

    # --- Data & Tokenization Parameters ---
    
    # Context length: The maximum number of tokens the model looks at
    # to predict the next token. (e.g., 256 characters)
    block_size = 256 
    
    # The dataset file path (used primarily for reference)
    data_filepath = "data/tiny_shakespeare.txt"
    
    # Vocabulary size will be determined dynamically by the dataset, 
    # but we initialize it here for clarity.
    vocab_size = None # To be set after loading the data (e.g., 65 characters)


    # --- Model Hyperparameters (GPT-style Architecture) ---

    # Embedding dimension (d_model in transformer terms)
    n_embed = 384
    
    # Number of Transformer blocks (layers)
    n_layer = 6
    
    # Number of attention heads (n_embed should be divisible by n_head)
    n_head = 6
    
    # Dropout rate for regularization (applied to embeddings, attention, and feed-forward)
    dropout = 0.2
    
    
    # --- Training Hyperparameters ---

    # Batch size: Number of independent sequences processed per forward/backward pass
    batch_size = 64
    
    # Number of training steps (or epochs)
    max_iters = 5000 
    
    # Initial learning rate 
    learning_rate = 3e-4
    
    # Interval for logging training status (loss, step count)
    log_interval = 500
    
    # Interval for evaluating the model on a separate validation set
    eval_interval = 500
    
    # Number of batches to use for validation
    eval_iters = 200
    
    # Device to run the training on (e.g., 'cuda' or 'cpu')
    device = DEVICE 
    
    
    # --- Checkpoint & Output Parameters ---
    
    # Directory to save model weights
    checkpoint_dir = "outputs/checkpoints"
    
    # Directory for training logs and loss curves
    log_dir = "outputs/logs"
    
    # Directory for generated text samples
    sample_dir = "outputs/samples"

    
# Optional: Print configuration upon import for quick verification
# print(f"--- Configuration Loaded ---")
# print(f"Device: {Config.device}")
# print(f"Model Parameters: L={Config.n_layer}, H={Config.n_head}, E={Config.n_embed}")
# print(f"Batch Size: {Config.batch_size}, Block Size: {Config.block_size}")
