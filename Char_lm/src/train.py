import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from pathlib import Path

# --- Import Core Modules ---
import sys
from pathlib import Path
# Adjust path to import modules from the project root 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import all dependencies
from src.config import Config
from src.dataset import CharDataset, train_data, val_data, VOCAB_SIZE, get_batch, encode, decode
from src.model import TransformerLanguageModel
from src.utils import set_seed, save_checkpoint, count_parameters, plot_loss_curve

# --- Initialization ---
# Set the seed for reproducibility
set_seed(42)

# Set the vocabulary size in Config, as it's critical for model initialization
Config.vocab_size = VOCAB_SIZE

# --- Model and Optimizer Setup ---
print(f"Initializing model on device: {Config.device}...")

# 1. Model initialization
model = TransformerLanguageModel()

# 2. Count parameters
n_params = count_parameters(model)
print(f"Total model parameters: {n_params / 1e6:.2f} Million")

# 3. Optimizer initialization (AdamW is standard for Transformers)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=Config.learning_rate
)

# --- Loss Estimation Function ---

@torch.no_grad()
def estimate_loss(model, data_split):
    """
    Performs quick estimation of the loss over eval_iters batches.
    """
    out = {}
    model.eval() # Set model to evaluation mode
    
    # Select the correct data split
    data_source = train_data if data_split == 'train' else val_data
    
    # Only run if data is large enough to form a dataset
    if len(data_source) < Config.block_size + 1:
        print(f"Warning: {data_split} data too small for evaluation.")
        model.train()
        return float('inf')

    # Use CharDataset and DataLoader for cleaner sampling logic
    dataset = CharDataset(data_source)
    # Using DataLoader without shuffle for evaluation is fine
    # Note: For simplicity and speed, we stick to the get_batch function for evaluation here.
    
    losses = torch.zeros(Config.eval_iters)
    for k in range(Config.eval_iters):
        # Sample a batch from the split
        X, Y = get_batch(data_split)
        
        # Move data to the correct device
        X, Y = X.to(Config.device), Y.to(Config.device)
        
        # Forward pass
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    
    out[data_split] = losses.mean().item()
    model.train() # Set model back to training mode
    return out[data_split]

# --- Training Loop ---

def train():
    """
    Runs the main training loop for the Transformer Language Model.
    """
    print(f"Starting training for {Config.max_iters} steps...")
    
    # Lists to store loss values for plotting
    train_loss_history = {}
    val_loss_history = {}
    
    start_time = time.time()
    
    for iter_step in range(1, Config.max_iters + 1):
        
        # 1. Periodic Evaluation and Checkpointing
        if iter_step % Config.eval_interval == 0:
            train_loss = estimate_loss(model, 'train')
            val_loss = estimate_loss(model, 'val')
            
            # Log and store losses
            print(f"Step {iter_step}/{Config.max_iters} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Time: {(time.time() - start_time):.2f}s")
            
            train_loss_history[iter_step] = train_loss
            val_loss_history[iter_step] = val_loss
            
            # Optional: Save checkpoint every N evaluation intervals
            if (iter_step // Config.eval_interval) % 5 == 0:
                 save_checkpoint(model, optimizer, iter_step)
        
        # 2. Text Generation Example
        if iter_step % (Config.eval_interval * 5) == 0:
            # Set model to evaluation mode for clean generation
            model.eval()
            
            # Create starting context (e.g., "ROMEO:\n")
            start_text = "ROMEO:\n"
            context = torch.tensor(encode(start_text), dtype=torch.long, device=Config.device).unsqueeze(0)
            
            generated_tokens = model.generate(context, max_new_tokens=200)
            generated_text = decode(generated_tokens[0].tolist())
            # --- Save Generated Sample to File ---
            sample_dir = Path(Config.sample_dir)
            sample_dir.mkdir(parents=True, exist_ok=True)
            # Corrected Line:
            sample_filename = sample_dir / f"sample_step_{iter_step:05d}.txt"

            with open(sample_filename, 'w', encoding='utf-8') as f:
                 f.write(generated_text)
    
            print(f"\n✅ Generated sample saved to: {sample_filename}")
            print("-" * 50)
            print(f"--- Sample Generation at Step {iter_step} ---")
            print(generated_text)
            print("-" * 50)
            
            # Set model back to training mode
            model.train()

        # --- Training Step ---

        # 3. Get Batch (x, y)
        xb, yb = get_batch('train')
        xb, yb = xb.to(Config.device), yb.to(Config.device)
        
        # 4. Forward Pass
        logits, loss = model(xb, yb)
        
        # 5. Backward Pass and Optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # 6. Log Training Loss (simple every 500 steps)
        if iter_step % Config.log_interval == 0 and iter_step % Config.eval_interval != 0:
             print(f"Step {iter_step}/{Config.max_iters} | Loss: {loss.item():.4f}")

    # --- Training Complete ---
    print("\nTraining complete.")
    
    # Final checkpoint save
    save_checkpoint(model, optimizer, Config.max_iters)
    
    # Final loss curve plot
    plot_loss_curve(train_loss_history, val_loss_history, Config.log_dir)

if __name__ == '__main__':
    train()