use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod cache;
pub mod ipfs;
pub mod scorer;
pub mod utils;

use cache::TinyLfuCache;
use ipfs::IpfsClient;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::OnceLock;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Global state — initialized once via `cerebro_init()`
// ---------------------------------------------------------------------------

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
static CACHE: OnceLock<TinyLfuCache<String>> = OnceLock::new();
static IPFS: OnceLock<IpfsClient> = OnceLock::new();

fn get_runtime() -> PyResult<&'static tokio::runtime::Runtime> {
    RUNTIME
        .get()
        .ok_or_else(|| PyRuntimeError::new_err("cerebro not initialized — call cerebro_init() first"))
}

fn get_cache() -> PyResult<&'static TinyLfuCache<String>> {
    CACHE
        .get()
        .ok_or_else(|| PyRuntimeError::new_err("cerebro not initialized — call cerebro_init() first"))
}

fn get_ipfs() -> PyResult<&'static IpfsClient> {
    IPFS.get()
        .ok_or_else(|| PyRuntimeError::new_err("cerebro not initialized — call cerebro_init() first"))
}

// ---------------------------------------------------------------------------
// PyO3 module
// ---------------------------------------------------------------------------

#[pymodule]
fn scorer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cerebro_init, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(cache_get, m)?)?;
    m.add_function(wrap_pyfunction!(cache_put, m)?)?;
    m.add_function(wrap_pyfunction!(cache_clear, m)?)?;
    m.add_function(wrap_pyfunction!(cache_len, m)?)?;
    m.add_function(wrap_pyfunction!(ipfs_fetch, m)?)?;
    m.add_function(wrap_pyfunction!(ipfs_pin, m)?)?;
    m.add_function(wrap_pyfunction!(ipfs_add_json, m)?)?;
    m.add_function(wrap_pyfunction!(ipfs_health, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

/// Initialize the CEREBRO scorer runtime.
///
/// Args:
///     ipfs_api: IPFS Kubo API URL (default: "http://127.0.0.1:5001")
///     cache_capacity: TinyLFU cache capacity (default: 10000)
#[pyfunction]
#[pyo3(signature = (ipfs_api="http://127.0.0.1:5001", cache_capacity=10000))]
fn cerebro_init(ipfs_api: &str, cache_capacity: usize) -> PyResult<()> {
    utils::init_tracing();

    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("cerebro-rt")
            .build()
            .expect("failed to create tokio runtime")
    });

    CACHE.get_or_init(|| TinyLfuCache::new(cache_capacity));
    IPFS.get_or_init(|| IpfsClient::new(ipfs_api));

    tracing::info!(
        ipfs_api,
        cache_capacity,
        "CEREBRO scorer initialized"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

/// Cosine similarity between two numpy arrays.
#[pyfunction]
fn cosine_similarity(
    py: Python<'_>,
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
) -> PyResult<f32> {
    py.allow_threads(|| {
        scorer::cosine_similarity(a.as_slice()?, b.as_slice()?)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// Dot product of two numpy arrays.
#[pyfunction]
fn dot_product(
    py: Python<'_>,
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
) -> PyResult<f32> {
    py.allow_threads(|| {
        scorer::dot_product(a.as_slice()?, b.as_slice()?)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// Batch cosine similarity.
///
/// Args:
///     queries: list of numpy arrays (f32)
///     docs: list of numpy arrays (f32)
///
/// Returns:
///     numpy array of similarity scores
#[pyfunction]
fn score_batch<'py>(
    py: Python<'py>,
    queries: Vec<PyReadonlyArray1<'py, f32>>,
    docs: Vec<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if queries.len() != docs.len() {
        return Err(PyValueError::new_err(format!(
            "batch size mismatch: {} queries vs {} docs",
            queries.len(),
            docs.len()
        )));
    }

    // Extract slices outside allow_threads (needs GIL)
    let q_vecs: Vec<Vec<f32>> = queries
        .iter()
        .map(|q| q.as_slice().map(|s| s.to_vec()))
        .collect::<std::result::Result<_, _>>()?;

    let d_vecs: Vec<Vec<f32>> = docs
        .iter()
        .map(|d| d.as_slice().map(|s| s.to_vec()))
        .collect::<std::result::Result<_, _>>()?;

    let scores = py.allow_threads(|| {
        let q_refs: Vec<&[f32]> = q_vecs.iter().map(|v| v.as_slice()).collect();
        let d_refs: Vec<&[f32]> = d_vecs.iter().map(|v| v.as_slice()).collect();
        scorer::score_batch(&q_refs, &d_refs)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(PyArray1::from_vec_bound(py, scores))
}

/// L2-normalize a numpy array in-place.
#[pyfunction]
fn normalize(mut a: numpy::PyReadwriteArray1<f32>) -> PyResult<()> {
    let slice = a.as_slice_mut()?;
    scorer::normalize(slice);
    Ok(())
}

// ---------------------------------------------------------------------------
// Cache functions
// ---------------------------------------------------------------------------

/// Get a value from the TinyLFU cache.
#[pyfunction]
fn cache_get(key: &str) -> PyResult<Option<String>> {
    Ok(get_cache()?.get(key))
}

/// Put a value into the TinyLFU cache.
///
/// Args:
///     key: cache key
///     value: cache value (string)
///     ttl_secs: optional TTL in seconds
#[pyfunction]
#[pyo3(signature = (key, value, ttl_secs=None))]
fn cache_put(key: String, value: String, ttl_secs: Option<u64>) -> PyResult<()> {
    let ttl = ttl_secs.map(Duration::from_secs);
    get_cache()?.put(key, value, ttl);
    Ok(())
}

/// Clear all entries from the cache.
#[pyfunction]
fn cache_clear() -> PyResult<()> {
    get_cache()?.clear();
    Ok(())
}

/// Get the number of entries in the cache.
#[pyfunction]
fn cache_len() -> PyResult<usize> {
    Ok(get_cache()?.len())
}

// ---------------------------------------------------------------------------
// IPFS functions
// ---------------------------------------------------------------------------

/// Fetch a CID from IPFS and save to disk.
///
/// Args:
///     cid: IPFS content identifier
///     output_path: local file path to write to
#[pyfunction]
fn ipfs_fetch(py: Python<'_>, cid: &str, output_path: &str) -> PyResult<()> {
    let rt = get_runtime()?;
    let ipfs = get_ipfs()?;
    let path = std::path::PathBuf::from(output_path);

    py.allow_threads(|| {
        rt.block_on(ipfs.fetch_cid(cid, &path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

/// Pin a CID on the local IPFS node.
#[pyfunction]
fn ipfs_pin(py: Python<'_>, cid: &str) -> PyResult<()> {
    let rt = get_runtime()?;
    let ipfs = get_ipfs()?;

    py.allow_threads(|| {
        rt.block_on(ipfs.pin_add(cid))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

/// Add JSON data to IPFS, returning the CID.
#[pyfunction]
fn ipfs_add_json(py: Python<'_>, data: &str) -> PyResult<String> {
    let rt = get_runtime()?;
    let ipfs = get_ipfs()?;

    let value: serde_json::Value = serde_json::from_str(data)
        .map_err(|e| PyValueError::new_err(format!("invalid JSON: {e}")))?;

    py.allow_threads(|| {
        rt.block_on(ipfs.add_json(&value))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

/// Check IPFS node health, returns peer ID string.
#[pyfunction]
fn ipfs_health(py: Python<'_>) -> PyResult<String> {
    let rt = get_runtime()?;
    let ipfs = get_ipfs()?;

    py.allow_threads(|| {
        let peer = rt
            .block_on(ipfs.id())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(peer.id)
    })
}
