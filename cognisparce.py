import numpy as np
from collections import defaultdict
import heapq

class CogniSparse:
    def __init__(self, dim: int = 768):
        """
        dim: embedding dimension (e.g. 768 for BERT-base)
        """
        self.dim = dim
        # One global zero vector shared by ALL words (huge memory saver)
        self.global_zero = np.zeros(dim, dtype=np.float32)
        
        # word → {'values': dict[pos → value], 'next_pos': int or None}
        self.word_data = defaultdict(lambda: {'values': {}, 'next_pos': None})
        
        # Lazy scheduler: min-heap of (next_pos, word) for pending words
        self.lazy_queue = []  # heapq: (next_pos, word)
        
        self.current_counter = 0

    def add_word(self, word: str, pos_values: dict[int, float]):
        """
        Add/update a word with its sparse position-value pairs.
        pos_values: {position: value} — only non-zero entries
        """
        data = self.word_data[word]
        data['values'].update(pos_values)
        
        # Update lazy scheduler if needed
        if pos_values:
            earliest = min(pos_values.keys())
            if data['next_pos'] is None or earliest < data['next_pos']:
                data['next_pos'] = earliest
                heapq.heappush(self.lazy_queue, (earliest, word))

    def get_vector(self, word: str, lazy: bool = False) -> np.ndarray:
        """
        Reconstruct full vector for a word.
        - lazy=False: full eager reconstruction (default)
        - lazy=True: only fill up to current counter (demand-driven)
        """
        if word not in self.word_data:
            return self.global_zero.copy()
        
        data = self.word_data[word]
        vec = self.global_zero.copy()
        
        if lazy:
            # Lazy fill: only positions <= current counter
            while data['next_pos'] is not None and data['next_pos'] <= self.current_counter:
                pos = data['next_pos']
                val = data['values'].pop(pos, 0.0)
                vec[pos] = val
                
                # Find next position (if any remain)
                remaining = [p for p in data['values'] if p > pos]
                if remaining:
                    data['next_pos'] = min(remaining)
                else:
                    data['next_pos'] = None
        else:
            # Eager: fill everything
            for pos, val in data['values'].items():
                vec[pos] = val
            data['next_pos'] = None  # done

        return vec

    def advance_counter(self, steps: int = 1):
        """Advance global counter (used in lazy mode)"""
        self.current_counter += steps
        
        # Process any words ready at or before current counter
        while self.lazy_queue and self.lazy_queue[0][0] <= self.current_counter:
            next_pos, word = heapq.heappop(self.lazy_queue)
            # Only process if still valid (avoid stale entries)
            if word in self.word_data and self.word_data[word]['next_pos'] == next_pos:
                self.get_vector(word, lazy=True)  # fills up to current counter

    def memory_usage_bytes(self):
        """Rough estimate of memory usage in bytes"""
        total = self.global_zero.nbytes  # global zero vector
        for data in self.word_data.values():
            # dict overhead + int/float pairs
            total += len(data['values']) * (8 + 8)  # pos (int64) + value (float64)
        return total

    def __len__(self):
        return len(self.word_data)

    def __str__(self):
        return f"CogniSparse(dim={self.dim}, words={len(self)})"


# Tiny demo / test

if __name__ == "__main__":
    cs = CogniSparse(dim=8)

    # Add some sparse words
    cs.add_word("hello", {0: 0.8, 3: -0.2, 7: 0.5})
    cs.add_word("world", {1: 0.9, 4: 0.3, 9: -0.1})
    cs.add_word("test", {2: 1.0, 5: 0.7})

    # Eager reconstruction
    print("Eager hello:", cs.get_vector("hello", lazy=False))

    # Lazy mode demo
    print("\nLazy mode:")
    for t in range(10):
        cs.advance_counter(1)
        vec = cs.get_vector("hello", lazy=True)
        print(f"Counter {t}: hello vector: {vec}")

    print(f"\nMemory usage estimate: {cs.memory_usage_bytes() / 1024:.2f} KB")