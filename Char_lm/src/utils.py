import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Import the Config class to ensure we know where to save/load checkpoints
try:
    from src.config import Config
except ImportError:
    print("Error: Could not import Config from src.config.")
    raise

# --- 1. Seeding Function ---

def set_seed(seed):
    """
    Sets the seed for reproducibility across multiple libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Set CUDA seeds
        torch.cuda.manual_seed_all(seed)
        # For determinism (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 2. Checkpoint Management ---

def save_checkpoint(model, optimizer, step, path=None):
    """
    Saves the model state, optimizer state, and current training step.
    
    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        step (int): The current training iteration count.
        path (str or Path, optional): Directory to save the checkpoint. Defaults to Config.checkpoint_dir.
    """
    # Use Config default path if none is provided
    if path is None:
        path = Config.checkpoint_dir
    
    # Ensure the directory exists
    Path(path).mkdir(parents=True, exist_ok=True)
    
    # Define the full filename
    filename = Path(path) / f"checkpoint_step_{step:05d}.pt"
    
    # Create the state dictionary
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': Config.vocab_size # Include essential config data
    }
    
    # Save the checkpoint
    torch.save(checkpoint, filename)
    print(f"-> Checkpoint saved to {filename}")


def load_checkpoint(path):
    """
    Loads a checkpoint and returns the state dictionary.
    
    Args:
        path (str or Path): Full path to the checkpoint file.
        
    Returns:
        dict: The loaded checkpoint dictionary containing model, optimizer, and step info.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")

    # Load the checkpoint dictionary
    checkpoint = torch.load(path, map_location=Config.device)
    
    # Update global Config if necessary (e.g., in a separate run script)
    if 'vocab_size' in checkpoint and checkpoint['vocab_size'] != Config.vocab_size:
        print(f"Warning: Loaded checkpoint has vocab size {checkpoint['vocab_size']}, but current Config is {Config.vocab_size}.")
        
    print(f"-> Checkpoint loaded from {path} (Step: {checkpoint['step']})")
    
    return checkpoint

# --- 3. Model Inspection ---

def count_parameters(model):
    """
    Counts the total number of trainable parameters in a model.
    
    Args:
        model (nn.Module): The model instance.
        
    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 4. Optional: Plotting ---

def plot_loss_curve(train_losses, val_losses, log_dir=None):
    """
    Generates and optionally saves a plot of the training and validation loss curves.

    Args:
        train_losses (list or dict): List of training loss values (or a dict where keys are steps).
        val_losses (list or dict): List of validation loss values (or a dict where keys are steps).
        log_dir (str or Path, optional): Directory to save the plot. Defaults to Config.log_dir.
    """
    if log_dir is None:
        log_dir = Config.log_dir
        
    # Determine the x-axis (steps)
    if isinstance(train_losses, dict):
        train_steps = list(train_losses.keys())
        train_values = list(train_losses.values())
    else:
        train_steps = list(range(len(train_losses)))
        train_values = train_losses

    if isinstance(val_losses, dict):
        val_steps = list(val_losses.keys())
        val_values = list(val_losses.values())
    else:
        val_steps = list(range(len(val_losses)))
        val_values = val_losses

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_values, label='Training Loss', color='blue')
    plt.plot(val_steps, val_values, label='Validation Loss', color='red')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(log_dir) / "loss_curve.png"
    plt.savefig(save_path)
    plt.close()
    print(f"-> Loss curve saved to {save_path}")