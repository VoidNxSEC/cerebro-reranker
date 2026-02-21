use crate::utils::{CerebroError, Result};
use serde::Deserialize;
use std::path::Path;
use std::time::Duration;
use tokio::io::AsyncWriteExt;

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Async HTTP client for the Kubo (IPFS) RPC API.
pub struct IpfsClient {
    http: reqwest::Client,
    base_url: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct AddResponse {
    hash: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct PinResponse {
    pins: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PeerId {
    #[serde(rename = "ID")]
    pub id: String,
    pub agent_version: Option<String>,
}

impl IpfsClient {
    /// Create a new client pointing at the Kubo HTTP API.
    ///
    /// `api_url` should be something like `http://127.0.0.1:5001`.
    pub fn new(api_url: &str) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(300)) // model downloads can be large
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(4)
            .build()
            .expect("failed to build reqwest client");

        Self {
            http,
            base_url: api_url.trim_end_matches('/').to_string(),
        }
    }

    /// Default client pointing at `http://127.0.0.1:5001`.
    pub fn default_local() -> Self {
        Self::new("http://127.0.0.1:5001")
    }

    // -----------------------------------------------------------------------
    // API methods
    // -----------------------------------------------------------------------

    /// Fetch CID content and write it to `output_path`.
    pub async fn fetch_cid(&self, cid: &str, output_path: &Path) -> Result<()> {
        let url = format!("{}/api/v0/cat?arg={}", self.base_url, cid);

        let resp = self
            .http
            .post(&url)
            .send()
            .await
            .map_err(|e| CerebroError::Ipfs(format!("fetch {cid}: {e}")))?;

        if !resp.status().is_success() {
            return Err(CerebroError::Ipfs(format!(
                "fetch {cid}: HTTP {}",
                resp.status()
            )));
        }

        // Stream to disk
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let mut file = tokio::fs::File::create(output_path).await?;
        let bytes = resp.bytes().await?;
        file.write_all(&bytes).await?;
        file.flush().await?;

        tracing::info!(cid, path = %output_path.display(), bytes = bytes.len(), "fetched from IPFS");
        Ok(())
    }

    /// Recursively fetch a CID (directory) to `output_dir`.
    pub async fn fetch_dir(&self, cid: &str, output_dir: &Path) -> Result<()> {
        let url = format!(
            "{}/api/v0/get?arg={}&archive=false",
            self.base_url, cid
        );

        let resp = self
            .http
            .post(&url)
            .send()
            .await
            .map_err(|e| CerebroError::Ipfs(format!("get {cid}: {e}")))?;

        if !resp.status().is_success() {
            return Err(CerebroError::Ipfs(format!(
                "get {cid}: HTTP {}",
                resp.status()
            )));
        }

        tokio::fs::create_dir_all(output_dir).await?;

        let bytes = resp.bytes().await?;
        let output_file = output_dir.join(cid);
        let mut file = tokio::fs::File::create(&output_file).await?;
        file.write_all(&bytes).await?;

        tracing::info!(cid, path = %output_dir.display(), "fetched directory from IPFS");
        Ok(())
    }

    /// Pin a CID so it won't be garbage collected.
    pub async fn pin_add(&self, cid: &str) -> Result<()> {
        let url = format!("{}/api/v0/pin/add?arg={}", self.base_url, cid);

        let resp = self
            .http
            .post(&url)
            .send()
            .await
            .map_err(|e| CerebroError::Ipfs(format!("pin {cid}: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CerebroError::Ipfs(format!(
                "pin {cid}: HTTP {status}: {body}"
            )));
        }

        let pin: PinResponse = resp.json().await?;
        tracing::info!(cid, pinned = ?pin.pins, "pinned to IPFS");
        Ok(())
    }

    /// Add JSON data to IPFS, returning the CID.
    pub async fn add_json(&self, data: &serde_json::Value) -> Result<String> {
        let url = format!("{}/api/v0/add", self.base_url);
        let payload = serde_json::to_vec(data)?;

        let part = reqwest::multipart::Part::bytes(payload)
            .file_name("data.json")
            .mime_str("application/json")
            .unwrap();

        let form = reqwest::multipart::Form::new().part("file", part);

        let resp = self
            .http
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| CerebroError::Ipfs(format!("add_json: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CerebroError::Ipfs(format!(
                "add_json: HTTP {status}: {body}"
            )));
        }

        let add: AddResponse = resp.json().await?;
        tracing::info!(cid = %add.hash, "added JSON to IPFS");
        Ok(add.hash)
    }

    /// Health check — returns the peer ID of the local node.
    pub async fn id(&self) -> Result<PeerId> {
        let url = format!("{}/api/v0/id", self.base_url);

        let resp = self
            .http
            .post(&url)
            .send()
            .await
            .map_err(|e| CerebroError::Ipfs(format!("id: {e}")))?;

        if !resp.status().is_success() {
            return Err(CerebroError::Ipfs(format!(
                "id: HTTP {}",
                resp.status()
            )));
        }

        let peer: PeerId = resp.json().await?;
        Ok(peer)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_construction() {
        let client = IpfsClient::new("http://127.0.0.1:5001");
        assert_eq!(client.base_url, "http://127.0.0.1:5001");
    }

    #[test]
    fn test_client_trims_trailing_slash() {
        let client = IpfsClient::new("http://127.0.0.1:5001/");
        assert_eq!(client.base_url, "http://127.0.0.1:5001");
    }

    #[test]
    fn test_default_local() {
        let client = IpfsClient::default_local();
        assert_eq!(client.base_url, "http://127.0.0.1:5001");
    }
}
