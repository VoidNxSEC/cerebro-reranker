use crate::utils::{CerebroError, Result};

// ---------------------------------------------------------------------------
// Public API — auto-dispatches to AVX2 or scalar at runtime
// ---------------------------------------------------------------------------

/// Cosine similarity between two f32 slices.
/// Returns value in [-1.0, 1.0].
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    check_dims(a, b)?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() >= 8 {
            // SAFETY: AVX2 confirmed via runtime check, length checked above.
            return Ok(unsafe { avx2::cosine_similarity_avx2(a, b) });
        }
    }

    Ok(scalar::cosine_similarity_scalar(a, b))
}

/// Dot product of two f32 slices.
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
    check_dims(a, b)?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() >= 8 {
            return Ok(unsafe { avx2::dot_product_avx2(a, b) });
        }
    }

    Ok(scalar::dot_product_scalar(a, b))
}

/// L2-normalize a vector in-place.
pub fn normalize(v: &mut [f32]) {
    let norm = scalar::l2_norm(v);
    if norm > f32::EPSILON {
        let inv = 1.0 / norm;
        v.iter_mut().for_each(|x| *x *= inv);
    }
}

/// Batch cosine similarity: scores[i] = cosine(queries[i], docs[i]).
/// Uses rayon for parallelism across pairs.
pub fn score_batch(queries: &[&[f32]], docs: &[&[f32]]) -> Result<Vec<f32>> {
    if queries.len() != docs.len() {
        return Err(CerebroError::Scoring(format!(
            "batch size mismatch: {} queries vs {} docs",
            queries.len(),
            docs.len()
        )));
    }

    use rayon::prelude::*;

    let results: std::result::Result<Vec<f32>, _> = queries
        .par_iter()
        .zip(docs.par_iter())
        .map(|(q, d)| cosine_similarity(q, d))
        .collect();

    results
}

// ---------------------------------------------------------------------------
// Dimension check
// ---------------------------------------------------------------------------

#[inline]
fn check_dims(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(CerebroError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(CerebroError::Scoring("empty vectors".into()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

mod scalar {
    pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    pub fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    pub fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
        let dot = dot_product_scalar(a, b);
        let norm_a = l2_norm(a);
        let norm_b = l2_norm(b);
        let denom = norm_a * norm_b;

        if denom < f32::EPSILON {
            return 0.0;
        }

        dot / denom
    }
}

// ---------------------------------------------------------------------------
// AVX2 implementation
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Horizontal sum of 8 packed f32 in a __m256.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_ps_avx(v: __m256) -> f32 {
        // [a0+a4, a1+a5, a2+a6, a3+a7]  (128-bit)
        let hi128 = _mm256_extractf128_ps(v, 1);
        let lo128 = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo128, hi128);
        // [s0+s2, s1+s3, ...]
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        // [s0+s1+s2+s3, ...]
        let shuf2 = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(final_sum)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        let mut sum = hsum_ps_avx(acc);

        // Handle remaining elements
        let tail_start = chunks * 8;
        for i in 0..remainder {
            sum += a[tail_start + i] * b[tail_start + i];
        }

        sum
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc_dot = _mm256_setzero_ps();
        let mut acc_a2 = _mm256_setzero_ps();
        let mut acc_b2 = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));

            acc_dot = _mm256_fmadd_ps(va, vb, acc_dot);
            acc_a2 = _mm256_fmadd_ps(va, va, acc_a2);
            acc_b2 = _mm256_fmadd_ps(vb, vb, acc_b2);
        }

        let mut dot = hsum_ps_avx(acc_dot);
        let mut norm_a2 = hsum_ps_avx(acc_a2);
        let mut norm_b2 = hsum_ps_avx(acc_b2);

        let tail_start = chunks * 8;
        for i in 0..remainder {
            let av = a[tail_start + i];
            let bv = b[tail_start + i];
            dot += av * bv;
            norm_a2 += av * av;
            norm_b2 += bv * bv;
        }

        let denom = (norm_a2 * norm_b2).sqrt();
        if denom < f32::EPSILON {
            return 0.0;
        }

        dot / denom
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&v, &v).unwrap();
        assert!(approx_eq(sim, 1.0), "got {sim}");
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(approx_eq(sim, 0.0), "got {sim}");
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(approx_eq(sim, -1.0), "got {sim}");
    }

    #[test]
    fn test_cosine_large_vector() {
        // 384-dim (MiniLM embedding size) — exercises AVX2 path
        let a: Vec<f32> = (0..384).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..384).map(|i| (i as f32).cos()).collect();

        let sim = cosine_similarity(&a, &b).unwrap();
        let scalar_sim = scalar::cosine_similarity_scalar(&a, &b);

        assert!(
            approx_eq(sim, scalar_sim),
            "SIMD ({sim}) != scalar ({scalar_sim})"
        );
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = dot_product(&a, &b).unwrap();
        assert!(approx_eq(dot, 32.0), "got {dot}");
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!(approx_eq(v[0], 0.6));
        assert!(approx_eq(v[1], 0.8));
    }

    #[test]
    fn test_score_batch() {
        let q1 = vec![1.0, 0.0, 0.0];
        let q2 = vec![0.0, 1.0, 0.0];
        let d1 = vec![1.0, 0.0, 0.0];
        let d2 = vec![0.0, 1.0, 0.0];

        let queries: Vec<&[f32]> = vec![&q1, &q2];
        let docs: Vec<&[f32]> = vec![&d1, &d2];

        let scores = score_batch(&queries, &docs).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(approx_eq(scores[0], 1.0));
        assert!(approx_eq(scores[1], 1.0));
    }
}
