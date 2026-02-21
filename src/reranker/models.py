"""
Model Registry and Management
IPFS-based model distribution
"""

import toml
from pathlib import Path
from typing import Dict, List, Optional
import structlog

log = structlog.get_logger()


class ModelRegistry:
    """
    Centralized model registry with IPFS support
    """

    def __init__(self, config_path: str = "/app/models.toml"):
        self.config_path = Path(config_path)
        self.models = {}
        self._load_registry()

    def _load_registry(self):
        """Load model configuration from TOML"""
        if not self.config_path.exists():
            log.warning("Model registry not found", path=str(self.config_path))
            return

        try:
            config = toml.load(self.config_path)
            self.models = config.get('models', {})

            log.info(
                "Model registry loaded",
                models=list(self.models.keys()),
                path=str(self.config_path)
            )

        except Exception as e:
            log.error("Failed to load model registry", error=str(e))

    def get_model(self, model_id: str) -> Dict:
        """Get model configuration"""
        if model_id == 'fast':
            # Map to actual fast model
            model_id = 'minilm'
        elif model_id == 'accurate':
            # Map to accurate model
            model_id = 'deberta'

        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")

        return self.models[model_id]

    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.models.keys())

    def get_all_models(self) -> Dict:
        """Get all model configurations"""
        return self.models

    def add_model(
        self,
        model_id: str,
        name: str,
        ipfs_cid: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Add a new model to registry"""
        self.models[model_id] = {
            'name': name,
            'ipfs_cid': ipfs_cid,
            **(metadata or {})
        }

        # Save to disk
        self._save_registry()

        log.info("Model added to registry", model_id=model_id, name=name)

    def update_model_cid(self, model_id: str, ipfs_cid: str):
        """Update IPFS CID for a model (after training)"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        self.models[model_id]['ipfs_cid'] = ipfs_cid
        self._save_registry()

        log.info(
            "Model CID updated",
            model_id=model_id,
            cid=ipfs_cid
        )

    def _save_registry(self):
        """Save registry to disk"""
        try:
            with open(self.config_path, 'w') as f:
                toml.dump({'models': self.models}, f)
        except Exception as e:
            log.error("Failed to save registry", error=str(e))
