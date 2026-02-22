import numpy as np
from collections import defaultdict
import heapq

class Sparsity:
    def __init__(self, dim: int = 768):
        """
        dim: embedding dimension (e.g. 768 for BERT-base)
        """
        self.dim = dim
        # Shared global zero vector (still float32 for output compatibility)
        self.global_zero = np.zeros(dim, dtype=np.float32)
        
        # word → {'values': dict[pos → int8], 'scale': float, 'next_pos': int or None}
        self.word_data = defaultdict(lambda: {'values': {}, 'scale': 1.0, 'next_pos': None})
        
        # Lazy scheduler queue
        self.lazy_queue = []  # heapq: (next_pos, word)
        
        self.current_counter = 0

    def add_word(self, word: str, pos_values: dict[int, float]):
        """
        Add/update a word with its sparse position-value pairs.
        Automatically quantizes floats to int8.
        """
        data = self.word_data[word]
        
        if not pos_values:
            return

        # Find max absolute value for scaling
        max_abs = max(abs(v) for v in pos_values.values()) + 1e-8
        scale = 127.0 / max_abs
        data['scale'] = scale
        
        for pos, val in pos_values.items():
            quantized = int(round(val * scale))
            quantized = np.clip(quantized, -128, 127)  # ensure int8 range
            data['values'][pos] = quantized
        
        # Update lazy scheduler
        if data['values']:
            earliest = min(data['values'].keys())
            if data['next_pos'] is None or earliest < data['next_pos']:
                data['next_pos'] = earliest
                heapq.heappush(self.lazy_queue, (earliest, word))

    def get_vector(self, word: str, lazy: bool = False) -> np.ndarray:
        """
        Reconstruct full vector (dequantizes int8 back to float32).
        """
        if word not in self.word_data:
            return self.global_zero.copy()
        
        data = self.word_data[word]
        vec = self.global_zero.copy()
        scale = data['scale']
        
        if lazy:
            while data['next_pos'] is not None and data['next_pos'] <= self.current_counter:
                pos = data['next_pos']
                quantized = data['values'].pop(pos, 0)
                vec[pos] = quantized / scale if scale != 0 else 0.0
                
                remaining = [p for p in data['values'] if p > pos]
                if remaining:
                    data['next_pos'] = min(remaining)
                else:
                    data['next_pos'] = None
        else:
            for pos, quantized in data['values'].items():
                vec[pos] = quantized / scale if scale != 0 else 0.0
            data['next_pos'] = None

        return vec

    def advance_counter(self, steps: int = 1):
        """Advance global counter and process ready words"""
        self.current_counter += steps
        
        while self.lazy_queue and self.lazy_queue[0][0] <= self.current_counter:
            next_pos, word = heapq.heappop(self.lazy_queue)
            if word in self.word_data and self.word_data[word]['next_pos'] == next_pos:
                self.get_vector(word, lazy=True)

    def memory_usage_bytes(self):
        """Estimate memory (positions as int64 + values as int8)"""
        total = self.global_zero.nbytes
        for data in self.word_data.values():
            total += len(data['values']) * (8 + 1)  # pos (int64) + value (int8)
        return total

    def __len__(self):
        return len(self.word_data)

    def __str__(self):
        return f"Sparsity(dim={self.dim}, words={len(self)})"


# Bigger demo to see real savings
if __name__ == "__main__":
    cs = Sparsity(dim=768)

    import random
    words = [f"w{i}" for i in range(10000)]  # 10k words
    for w in words:
        num_nonzeros = random.randint(8, 32)
        positions = random.sample(range(768), num_nonzeros)
        values = np.random.randn(num_nonzeros) * 0.1  # realistic small values
        cs.add_word(w, dict(zip(positions, values)))

    print(f"Words: {len(cs)}")
    print(f"Memory estimate: {cs.memory_usage_bytes() / 1024 / 1024:.2f} MB")

    dense_size = len(cs) * 768 * 4 / 1024 / 1024  # float32
    print(f"Dense would be ~{dense_size:.2f} MB")
    print(f"Savings factor: ~{dense_size / (cs.memory_usage_bytes() / 1024 / 1024):.1f}×")