# Sparsity

Sparse embeddings with lazy on-demand reconstruction — 10–30× smaller than dense for local/edge AI.

## Why
- Dense embeddings waste 90–99% space on zeros  
- Sparsity stores only non-zero positions + values  
- Shared global zero vector  
- Lazy scheduler: fill only when needed (heap-free)

## Quick start
```python
cs = Sparsity(dim=768)
cs.add_word("hello", {0: 0.8, 3: -0.2, 7: 0.5})
vec = cs.get_vector("hello", lazy=True)   # on-demand
cs.advance_counter(10)                    # move clock forward