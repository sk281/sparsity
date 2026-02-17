# Cogni-Sparse

Sparse embeddings with lazy, on-demand reconstruction.

**Core idea**  
Each word stores only non-zero position-value pairs (dict or sorted list).  
One global shared zero-vector (size n) for all words.  
Reconstruction: copy zero vector + scatter sparse values.

**Lazy scheduler** (optional)  
Global counter + per-word `next_pos` variable.  
Only fill a position when counter reaches it — demand-driven, heap-free.

**Benefits**  
- 10–30× smaller than dense (often 10–15×)  
- Fast reconstruction (O(k) scatter, k usually 8–64)  
- Incremental: add words easily  
- Interpretable: see exactly which dimensions are active  

**Limitations**  
- Slower batched GPU matmul without sparse kernels  
- Training updates slightly slower  
- Personal prototype — no large-scale benchmarks yet  

MIT license. Built from intuition, numbers vary. If you want to extend it (benchmarks, sparse matmul, etc.) — fork/PR away.