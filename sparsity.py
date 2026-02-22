import numpy as np
import time
import random
from collections import defaultdict

class SparsityDebt:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.global_zero = np.zeros(dim, dtype=np.float32)
        
        # word → {'values': dict[pos → int8], 'scale': float, 'future_positions': sorted list of remaining pos}
        self.word_data = defaultdict(lambda: {'values': {}, 'scale': 1.0, 'future_positions': []})
        
        self.current_step = 0

    def add_word(self, word: str, pos_values: dict[int, float]):
        """
        Add positions + values. Quantizes to int8. Adds and sorts future positions.
        """
        data = self.word_data[word]
        
        if not pos_values:
            return

        # Quantize values
        max_abs = max(abs(v) for v in pos_values.values()) + 1e-8
        scale = 127.0 / max_abs
        data['scale'] = scale
        
        for pos, val in pos_values.items():
            if not (0 <= pos < self.dim):
                print(f"Warning: skipping invalid position {pos} (dim={self.dim})")
                continue
            quantized = int(round(val * scale))
            quantized = np.clip(quantized, -128, 127)
            data['values'][pos] = quantized
        
        # Add and sort future positions
        new_positions = sorted(pos_values.keys())
        current_set = set(data['future_positions'])
        data['future_positions'] = sorted(current_set.union(new_positions))

    def get_next_word(self):
        """
        Return the word whose NEXT position is the soonest (smallest gap).
        Tiebreaker: word with most remaining positions.
        """
        if not self.word_data:
            return None

        earliest_gap = float('inf')
        candidates = []

        for word, data in self.word_data.items():
            if not data['future_positions']:
                continue
            next_pos = data['future_positions'][0]
            gap = next_pos - self.current_step
            if gap < 0:
                print(f"Warning: past position {next_pos} for '{word}' — skipping")
                data['future_positions'].pop(0)
                continue
            if gap < earliest_gap:
                earliest_gap = gap
                candidates = [(word, len(data['future_positions']))]
            elif gap == earliest_gap:
                candidates.append((word, len(data['future_positions'])))

        if not candidates:
            return None

        # Tiebreaker: most remaining positions
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    def advance_to_next(self):
        """
        Place the next word (remove its earliest position).
        Jump current_step to just after the placed position.
        Returns (word, placed_position) or None.
        """
        word = self.get_next_word()
        if not word:
            return None

        data = self.word_data[word]
        pos = data['future_positions'].pop(0)  # remove earliest
        
        # Jump current_step to after this position
        self.current_step = pos + 1
        
        # Clean up if no more positions
        if not data['future_positions']:
            del self.word_data[word]

        return word, pos

    def get_vector(self, word: str) -> np.ndarray:
        if word not in self.word_data:
            return self.global_zero.copy()

        data = self.word_data[word]
        vec = self.global_zero.copy()
        scale = data['scale']

        for pos, quantized in data['values'].items():
            if 0 <= pos < self.dim:
                vec[pos] = quantized / scale if scale != 0 else 0.0

        return vec

    def memory_usage_bytes(self):
        total = self.global_zero.nbytes
        for data in self.word_data.values():
            total += len(data['values']) * (8 + 1)  # pos int64 + value int8
        return total

    def __len__(self):
        return len(self.word_data)


# Realistic benchmark
if __name__ == "__main__":
    dim = 768

    # Text repeated to have meaningful frequencies
    text = """
    The quick brown fox The over the lazy dog. This is a test sentence to see word frequencies.
    Sparsity is a memory efficient way to store embeddings. Lazy reconstruction saves time and space.
    In real applications, common words appear many times while rare words appear few times.
    This creates natural differences that the scheduler can exploit.
    """ * 150

    words = text.split()
    print(f"Total tokens: {len(words)}")

    # Build word → list of positions
    word_positions = defaultdict(list)
    for pos, word in enumerate(words):
        word_positions[word].append(pos)

    print(f"Unique words: {len(word_positions)}")

    sd = SparsityDebt(dim=dim)

    start_time = time.perf_counter()

    for word, positions in word_positions.items():
        values = {p: 1.0 for p in positions}
        sd.add_word(word, values)

    add_time = time.perf_counter() - start_time
    print(f"Added all positions in {add_time:.3f} seconds")

    sparse_mem = sd.memory_usage_bytes() / 1024 / 1024
    dense_mem_estimate = len(word_positions) * dim * 4 / 1024 / 1024
    savings = dense_mem_estimate / sparse_mem if sparse_mem > 0 else float('inf')

    print(f"Sparse memory: {sparse_mem:.2f} MB")
    print(f"Dense estimate: {dense_mem_estimate:.2f} MB")
    print(f"Savings factor: ~{savings:.1f}×")

    # Speed test
    print("\nReconstructing 100 random words...")
    start_time = time.perf_counter()

    word_list = list(word_positions.keys())
    for _ in range(100):
        word = random.choice(word_list)
        vec = sd.get_vector(word)

    recon_time = time.perf_counter() - start_time
    print(f"Reconstructed 100 vectors in {recon_time:.3f} seconds")
    print(f"Average per vector: {recon_time / 100 * 1000:.2f} ms")

    # Simulate scheduler (advance until we placed 20 words or no more)
    print("\nSimulating scheduler (advance to next real position):")
    placement_count = 0
    while placement_count < 20:
        result = sd.advance_to_next()
        if result:
            word, pos = result
            print(f"Step {sd.current_step}: Placed '{word}' at position {pos}")
            placement_count += 1
        else:
            print("No more placements needed")
            break