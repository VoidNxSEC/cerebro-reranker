#!/usr/bin/env python3
# scripts/compare_performance.py

import time
import numpy as np
from sentence_transformers import CrossEncoder
import scorer  # Rust FFI

def benchmark(func, *args, iterations=100):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return np.mean(times) * 1000, np.std(times) * 1000

# Setup
query_emb = np.random.rand(384).astype(np.float32).tolist()
doc_embs = [np.random.rand(384).astype(np.float32).tolist() for _ in range(1000)]

# Python baseline
def python_cosine(q, docs):
    scores = []
    for d in docs:
        dot = sum(a*b for a, b in zip(q, d))
        q_norm = sum(a*a for a in q) ** 0.5
        d_norm = sum(b*b for b in d) ** 0.5
        scores.append(dot / (q_norm * d_norm))
    return scores

# NumPy
def numpy_cosine(q, docs):
    q = np.array(q)
    docs = np.array(docs)
    dots = docs @ q
    q_norm = np.linalg.norm(q)
    doc_norms = np.linalg.norm(docs, axis=1)
    return (dots / (q_norm * doc_norms)).tolist()

print("🚀 Performance Comparison\n")

# Benchmark
python_time, python_std = benchmark(python_cosine, query_emb, doc_embs)
print(f"Python:    {python_time:.2f}ms ± {python_std:.2f}ms")

numpy_time, numpy_std = benchmark(numpy_cosine, query_emb, doc_embs)
print(f"NumPy:     {numpy_time:.2f}ms ± {numpy_std:.2f}ms")

rust_time, rust_std = benchmark(scorer.batch_score, query_emb, doc_embs)
print(f"Rust SIMD: {rust_time:.2f}ms ± {rust_std:.2f}ms")

print(f"\n🔥 Speedup:")
print(f"  vs Python: {python_time/rust_time:.2f}x")
print(f"  vs NumPy:  {numpy_time/rust_time:.2f}x")
