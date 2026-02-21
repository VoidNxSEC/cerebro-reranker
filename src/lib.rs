//! CEREBRO Scorer - High-performance FFI library
//!
//! Provides SIMD-accelerated scoring for reranking with Python bindings

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod scorer;
mod ipfs;
mod cache;
mod utils;

pub use scorer::*;
pub use ipfs::*;
pub use cache::*;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Fast batch scoring using SIMD
#[pyfunction]
fn batch_score(
    py: Python<'_>,
    query_embedding: Vec<f32>,
    doc_embeddings: Vec<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| {
        scorer::batch_cosine_similarity(&query_embedding, &doc_embeddings)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}

/// Fast top-k selection without full sort
#[pyfunction]
fn top_k_indices(scores: Vec<f32>, k: usize) -> PyResult<Vec<usize>> {
    Ok(scorer::top_k_indices(&scores, k))
}

/// IPFS operations
#[pyfunction]
fn ipfs_pin(cid: String) -> PyResult<bool> {
    pyo3_asyncio::tokio::get_runtime().block_on(async {
        ipfs::pin_content(&cid)
            .await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}

#[pyfunction]
fn ipfs_fetch(cid: String, output_path: String) -> PyResult<()> {
    pyo3_asyncio::tokio::get_runtime().block_on(async {
        ipfs::fetch_content(&cid, &output_path)
            .await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}

/// Module initialization
#[pymodule]
fn scorer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_score, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_indices, m)?)?;
    m.add_function(wrap_pyfunction!(ipfs_pin, m)?)?;
    m.add_function(wrap_pyfunction!(ipfs_fetch, m)?)?;

    Ok(())
}
