import numpy as np
import time
import random
from collections import defaultdict, deque

class SparsityDebtCorrect:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.global_zero = np.zeros(dim, dtype=np.float32)
        
        # word → {'values': dict[pos → int8], 'scale': float, 'future_positions': sorted list or deque}
        self.word_data = defaultdict(lambda: {'values': {}, 'scale': 1.0, 'future_positions': deque()})
        
        self.current_step = 0

    def add_word(self, word: str, pos_values: dict[int, float]):
        data = self.word_data[word]
        
        if not pos_values:
            return

        max_abs = max(abs(v) for v in pos_values.values()) + 1e-8
        scale = 127.0 / max_abs
        data['scale'] = scale
        
        for pos, val in pos_values.items():
            if not (0 <= pos < self.dim):
                continue
            quantized = int(round(val * scale))
            quantized = np.clip(quantized, -128, 127)
            data['values'][pos] = quantized
        
        # Add and sort positions
        new_pos = sorted(pos_values.keys())
        current = set(data['future_positions'])
        data['future_positions'] = deque(sorted(current.union(new_pos)))

    def get_next_word(self):
        if not self.word_data:
            return None

        min_gap = float('inf')
        candidates = []

        for word, data in self.word_data.items():
            if not data['future_positions']:
                continue
            next_pos = data['future_positions'][0]
            gap = next_pos - self.current_step
            if gap < 0:
                # Past position — skip (shouldn't happen if jumped correctly)
                data['future_positions'].popleft()
                continue
            if gap < min_gap:
                min_gap = gap
                candidates = [(word, len(data['future_positions']))]
            elif gap == min_gap:
                candidates.append((word, len(data['future_positions'])))

        if not candidates:
            return None

        # Tiebreaker: most remaining positions
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    def advance_to_next(self):
        word = self.get_next_word()
        if not word:
            return None

        data = self.word_data[word]
        pos = data['future_positions'].popleft()
        
        # Jump current_step to after this placement
        self.current_step = pos + 1
        
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
            total += len(data['values']) * (8 + 1)
        return total


# Realistic benchmark with correct distance scheduling
if __name__ == "__main__":
    dim = 768

    text = """
    The quick brown fox jumps over the lazy dog. This is a test sentence to see word frequencies.
    Sparsity is a memory efficient way to store embeddings. Lazy reconstruction saves time and space.
    In real applications, common words appear many times while rare words appear few times.
    This creates natural debt differences that the hungry-word scheduler can exploit.
    """ * 150

    words = text.split()
    print(f"Total tokens: {len(words)}")

    word_positions = defaultdict(list)
    for pos, word in enumerate(words):
        word_positions[word].append(pos)

    print(f"Unique words: {len(word_positions)}")

    sd = SparsityDebtCorrect(dim=dim)

    start_time = time.time()

    for word, positions in word_positions.items():
        values = {p: 1.0 for p in positions}
        sd.add_word(word, values)

    add_time = time.time() - start_time
    print(f"Added all positions in {add_time:.3f} seconds")

    sparse_mem = sd.memory_usage_bytes() / 1024 / 1024
    dense_mem_estimate = len(word_positions) * dim * 4 / 1024 / 1024
    savings = dense_mem_estimate / sparse_mem if sparse_mem > 0 else float('inf')

    print(f"Sparse memory: {sparse_mem:.2f} MB")
    print(f"Dense estimate: {dense_mem_estimate:.2f} MB")
    print(f"Savings factor: ~{savings:.1f}×")

    # Speed test
    print("\nReconstructing 100 random words...")
    start_time = time.time()

    word_list = list(word_positions.keys())
    for _ in range(100):
        word = random.choice(word_list)
        vec = sd.get_vector(word)

    recon_time = time.time() - start_time
    print(f"Reconstructed 100 vectors in {recon_time:.3f} seconds")
    print(f"Average per vector: {recon_time / 100 * 1000:.2f} ms")

    # Simulate correct distance scheduler
    print("\nSimulating correct distance-based scheduler (advance to next real position):")
    placement_count = 0
    max_placements = 50  # limit demo
    while placement_count < max_placements:
        result = sd.advance_to_next()
        if result:
            word, pos = result
            print(f"Step {sd.current_step}: Placed '{word}' at real position {pos}")
            placement_count += 1
        else:
            print("No more placements needed")
            break