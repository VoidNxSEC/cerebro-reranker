//! IPFS client for model distribution

use anyhow::{Result, Context};
use ipfs_api_backend_hyper::{IpfsApi, IpfsClient, TryFromUri};
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

/// Pin content to IPFS
pub async fn pin_content(cid: &str) -> Result<bool> {
    let client = IpfsClient::from_str("http://127.0.0.1:5001")?;

    info!("Pinning CID: {}", cid);

    client
        .pin_add(cid, true)
        .await
        .context("Failed to pin content")?;

    info!("Successfully pinned: {}", cid);
    Ok(true)
}

/// Fetch content from IPFS
pub async fn fetch_content(cid: &str, output_path: &str) -> Result<()> {
    let client = IpfsClient::from_str("http://127.0.0.1:5001")?;

    info!("Fetching CID: {} to {}", cid, output_path);

    let data = client
        .cat(cid)
        .map_ok(|chunk| chunk.to_vec())
        .try_concat()
        .await
        .context("Failed to fetch from IPFS")?;

    // Write to file
    let mut file = File::create(output_path).await?;
    file.write_all(&data).await?;
    file.flush().await?;

    info!("Successfully fetched {} bytes", data.len());
    Ok(())
}

/// Add content to IPFS and return CID
pub async fn add_content(file_path: &str) -> Result<String> {
    let client = IpfsClient::from_str("http://127.0.0.1:5001")?;

    info!("Adding file to IPFS: {}", file_path);

    let file = tokio::fs::read(file_path).await?;

    let response = client
        .add(std::io::Cursor::new(file))
        .await
        .context("Failed to add to IPFS")?;

    let cid = response.hash;
    info!("Added to IPFS with CID: {}", cid);

    // Auto-pin
    pin_content(&cid).await?;

    Ok(cid)
}

/// Check if content is pinned
pub async fn is_pinned(cid: &str) -> Result<bool> {
    let client = IpfsClient::from_str("http://127.0.0.1:5001")?;

    let pins = client.pin_ls(Some(cid), None).await?;

    Ok(pins.keys.contains_key(cid))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires IPFS daemon
    async fn test_ipfs_operations() {
        // Create test file
        let test_data = b"test content";
        tokio::fs::write("/tmp/test_ipfs.txt", test_data).await.unwrap();

        // Add to IPFS
        let cid = add_content("/tmp/test_ipfs.txt").await.unwrap();

        // Check pinned
        assert!(is_pinned(&cid).await.unwrap());

        // Fetch back
        fetch_content(&cid, "/tmp/test_ipfs_fetched.txt").await.unwrap();

        let fetched = tokio::fs::read("/tmp/test_ipfs_fetched.txt").await.unwrap();
        assert_eq!(fetched, test_data);
    }
}
