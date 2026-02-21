//! SIMD-accelerated scoring engine
//!
//! Uses platform-specific SIMD instructions for maximum performance

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::arch::x86_64::*;

/// Batch cosine similarity with SIMD acceleration
pub fn batch_cosine_similarity(
    query: &[f32],
    documents: &[Vec<f32>],
) -> Result<Vec<f32>> {
    if query.is_empty() {
        return Err(anyhow!("Query embedding is empty"));
    }

    // Normalize query once
    let query_norm = l2_norm(query);

    // Parallel scoring
    let scores: Vec<f32> = documents
        .par_iter()
        .map(|doc| {
            if doc.len() != query.len() {
                0.0  // Invalid dimension
            } else {
                cosine_similarity_simd(query, doc, query_norm)
            }
        })
        .collect();

    Ok(scores)
}

/// SIMD-accelerated cosine similarity
#[inline]
fn cosine_similarity_simd(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { cosine_similarity_avx2(a, b, a_norm) }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { cosine_similarity_sse41(a, b, a_norm) }
        } else {
            cosine_similarity_scalar(a, b, a_norm)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        cosine_similarity_scalar(a, b, a_norm)
    }
}

/// AVX2 implementation (8 floats at a time)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let len = a.len();
    let mut dot_product = _mm256_setzero_ps();
    let mut b_norm_sq = _mm256_setzero_ps();

    let mut i = 0;

    // Process 8 floats at a time
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));

        // dot_product += a * b (using FMA)
        dot_product = _mm256_fmadd_ps(va, vb, dot_product);

        // b_norm_sq += b * b
        b_norm_sq = _mm256_fmadd_ps(vb, vb, b_norm_sq);

        i += 8;
    }

    // Horizontal sum
    let mut dot_sum = horizontal_sum_avx2(dot_product);
    let mut b_norm_sum = horizontal_sum_avx2(b_norm_sq);

    // Handle remaining elements
    while i < len {
        dot_sum += a[i] * b[i];
        b_norm_sum += b[i] * b[i];
        i += 1;
    }

    // Cosine similarity
    let b_norm = b_norm_sum.sqrt();
    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        dot_sum / (a_norm * b_norm)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Sum across 8 lanes
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);

    let hi64 = _mm_unpackhi_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);

    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x1);
    let sum = _mm_add_ss(sum64, hi32);

    _mm_cvtss_f32(sum)
}

/// SSE4.1 implementation (4 floats at a time)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn cosine_similarity_sse41(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let len = a.len();
    let mut dot_product = _mm_setzero_ps();
    let mut b_norm_sq = _mm_setzero_ps();

    let mut i = 0;

    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));

        let prod = _mm_mul_ps(va, vb);
        dot_product = _mm_add_ps(dot_product, prod);

        let b_sq = _mm_mul_ps(vb, vb);
        b_norm_sq = _mm_add_ps(b_norm_sq, b_sq);

        i += 4;
    }

    let mut dot_sum = horizontal_sum_sse(dot_product);
    let mut b_norm_sum = horizontal_sum_sse(b_norm_sq);

    while i < len {
        dot_sum += a[i] * b[i];
        b_norm_sum += b[i] * b[i];
        i += 1;
    }

    let b_norm = b_norm_sum.sqrt();
    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        dot_sum / (a_norm * b_norm)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn horizontal_sum_sse(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v);
    let sums = _mm_add_ps(v, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

/// Scalar fallback
#[inline]
fn cosine_similarity_scalar(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let b_norm: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        dot / (a_norm * b_norm)
    }
}

/// L2 norm
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Fast top-k selection using partial sort
pub fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    if k >= scores.len() {
        // Return all indices sorted
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        return indices;
    }

    // Partial heap-based selection (faster than full sort)
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    #[derive(PartialEq)]
    struct IndexedScore {
        index: usize,
        score: f32,
    }

    impl Eq for IndexedScore {}

    impl PartialOrd for IndexedScore {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            // Reverse order for max-heap
            other.score.partial_cmp(&self.score)
        }
    }

    impl Ord for IndexedScore {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap_or(Ordering::Equal)
        }
    }

    let mut heap = BinaryHeap::with_capacity(k + 1);

    for (idx, &score) in scores.iter().enumerate() {
        heap.push(IndexedScore { index: idx, score });
        if heap.len() > k {
            heap.pop();
        }
    }

    let mut result: Vec<usize> = heap.iter().map(|x| x.index).collect();
    result.sort_unstable_by(|&a, &b| {
        scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal)
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let a_norm = l2_norm(&a);
        let score = cosine_similarity_scalar(&a, &b, a_norm);

        assert!(score > 0.9 && score <= 1.0);
    }

    #[test]
    fn test_top_k() {
        let scores = vec![0.5, 0.9, 0.1, 0.7, 0.3];
        let top = top_k_indices(&scores, 3);

        assert_eq!(top, vec![1, 3, 0]);
    }
}
