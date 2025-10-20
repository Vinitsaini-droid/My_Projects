import torch
import torch.nn.functional as F
import argparse
import sys
from pathlib import Path

# Adjust path to import modules from the project root 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import all dependencies
from src.config import Config
from src.dataset import VOCAB_SIZE, encode, decode, BLOCK_SIZE
from src.model import TransformerLanguageModel
from src.utils import load_checkpoint

# --- Setup Configuration ---
# Set the vocabulary size in Config, as it's needed for model initialization
Config.vocab_size = VOCAB_SIZE
# Update BLOCK_SIZE in case it was modified in Config outside of model.py
Config.block_size = BLOCK_SIZE 

@torch.no_grad()
def generate_text(model, start_text, max_new_tokens, temperature):
    """
    Generates text autoregressively using the trained model.

    Args:
        model (TransformerLanguageModel): The loaded trained model.
        start_text (str): The initial context string.
        max_new_tokens (int): Maximum tokens to generate.
        temperature (float): Controls randomness (higher = more random).

    Returns:
        str: The generated text.
    """
    model.eval() # Set model to evaluation mode

    # 1. Encode the starting text and move to device
    try:
        context_tokens = encode(start_text)
    except KeyError as e:
        # Handle characters not in the vocabulary
        print(f"Error: Start text contains character not in vocabulary: {e}")
        return start_text

    idx = torch.tensor(context_tokens, dtype=torch.long, device=Config.device).unsqueeze(0) # (1, T_start)

    print(f"Generating {max_new_tokens} tokens with temperature {temperature:.2f}...")

    # 2. Iterative Generation Loop
    for _ in range(max_new_tokens):
        
        # Crop context to BLOCK_SIZE for model input
        idx_cond = idx[:, -Config.block_size:]
        
        # Get predictions (logits)
        logits, _ = model(idx_cond)
        
        # Focus on the last time step prediction: (1, VOCAB_SIZE)
        logits = logits[:, -1, :] 
        
        # Apply temperature scaling
        logits = logits / temperature
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) 
        
        # Sample from the distribution (multinomial sampling)
        idx_next = torch.multinomial(probs, num_samples=1) # (1, 1)
        
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (1, T+1)
        
    # 3. Decode and Return
    generated_tokens_list = idx[0].tolist()
    return decode(generated_tokens_list)


def main():
    """
    Main function to handle command-line arguments and run generation.
    """
    parser = argparse.ArgumentParser(description="Generate text using a trained Mini-Transformer model.")
    parser.add_argument("--checkpoint", type=str, required= True,
                        help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--start_text", type=str, default="The ",
                        help="Starting prompt for text generation.")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.0 to 1.0+). Lower is less random.")
    
    args = parser.parse_args()

    # --- Load Model and Weights ---
    
    # 1. Instantiate the model architecture
    model = TransformerLanguageModel()

    # 2. Load weights
    try:
        checkpoint = load_checkpoint(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
        print("Ensure the checkpoint path is correct and the model architecture matches the saved state.")
        sys.exit(1)
    
    model.to(Config.device)
    
    # --- Run Generation ---
    
    # Clean up temperature value
    temperature = max(0.1, args.temperature) 

    # Generate the output
    output_text = generate_text(model, args.start_text, args.max_tokens, temperature)
    
    print("\n" + "=" * 60)
    print("           ✨ GENERATED TEXT ✨")
    print("=" * 60)
    print(output_text)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()