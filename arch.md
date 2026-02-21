cerebro-reranker/
├── flake.nix # Entry point
├── flake.lock
├── modules/
│ ├── reranker-service.nix # Core service
│ ├── ipfs-cluster.nix # Distributed model serving
│ ├── training-pipeline.nix # GCP training automation
│ ├── cache-layer.nix # Redis + IPFS cache
│ └── monitoring.nix # Observability stack
├── src/
│ ├── reranker/
│ │ ├── server.py # FastAPI server
│ │ ├── hybrid_engine.py # Hybrid reranking logic
│ │ ├── models.py # Model management
│ │ └── cache.py # IPFS-backed caching
│ ├── training/
│ │ ├── train.py # Fine-tuning pipeline
│ │ └── export.py # ONNX export + quantization
│ └── lib/
│ ├── scorer.rs # Rust FFI for fast scoring
│ └── ipfs_client.rs # IPFS pinning client
├── configs/
│ ├── models.toml # Model registry
│ └── gcp-training.yaml # Vertex AI config
└── scripts/
├── setup-gcp.sh
└── deploy.sh
