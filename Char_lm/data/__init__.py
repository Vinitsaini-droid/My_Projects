from pathlib import Path

# Define the root path of the data directory relative to this __init__.py file.
# This assumes the project is run from the root or the environment path is set correctly.
DATA_DIR = Path(__file__).parent

# Define the full path to the raw dataset file
DATASET = DATA_DIR / 'tiny_shakespeare.txt'

# You can optionally define the file path to be imported easily elsewhere.
# Example usage in src/dataset.py:
# from data import DATASET 
# with open(DATASET, 'r') as f:
#     text = f.read()

# You could also add a function to check if the file exists:
# def check_data_exists():
#     return DATASET.exists()

print(f"Data package initialized. Dataset path set to: {DATASET}")
