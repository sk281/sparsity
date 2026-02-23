# Sparsity

Sparse embeddings with debt-line lazy reconstruction — 15–50× smaller than dense for local / edge / on-device AI.

## What it is
A simple, from-scratch sparse vector system I built from first principles.  

Key features / my twist:
- Stores **only non-zero positions** + int8 quantized values (big memory win)
- Shared global zero vector (no duplication)
- **Debt-line scheduler**: debt = remaining positions per word  
  - Always places the word with highest debt  
  - Placing reduces its debt by 1, increases everyone else's debt by 1  
  - No heap, no priority queue — priority emerges naturally from debt
- Lazy reconstruction: `get_vector()` pulls values on demand
- Built-in int8 quantization (automatic scaling)

## Why bother
Dense embeddings waste 90–99% space on zeros.  
This is a hack to make local LLMs (phones, laptops, RTX cards) run longer context / bigger models without OOM.  
No fancy libraries — pure NumPy + Python.

## Quick start
```python
from sparsity_debt import SparsityDebt  # or whatever you named the file

sd = SparsityDebt(dim=768)

# Add sparse positions + values (auto-quantized to int8)
sd.add_word("hello", {0: 0.8, 3: -0.2, 7: 0.5})

# Get full dequantized vector
vec = sd.get_vector("hello")
print(vec.shape)  # (768,)

# Simulate debt-line placement (optional)
next_word = sd.place_next()
print(next_word)