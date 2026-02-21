"""Tests for ModelRegistry."""

import pytest
from models import ModelRegistry


class TestModelRegistry:
    def test_load_registry(self, model_registry):
        assert len(model_registry.models) == 4
        assert "minilm" in model_registry.models
        assert "deberta" in model_registry.models

    def test_list_models(self, model_registry):
        models = model_registry.list_models()
        assert "minilm" in models
        assert "electra" in models
        assert "deberta" in models
        assert "custom" in models

    def test_get_model_direct(self, model_registry):
        model = model_registry.get_model("minilm")
        assert model["name"] == "ms-marco-MiniLM-L-6-v2"
        assert model["size_mb"] == 80
        assert model["accuracy"] == 0.89
        assert model["ipfs_cid"].startswith("bafybei")

    def test_get_model_fast_alias(self, model_registry):
        """'fast' should map to minilm."""
        model = model_registry.get_model("fast")
        assert model["name"] == "ms-marco-MiniLM-L-6-v2"

    def test_get_model_accurate_alias(self, model_registry):
        """'accurate' should map to deberta."""
        model = model_registry.get_model("accurate")
        assert model["name"] == "ms-marco-deberta-v3-base"

    def test_get_model_not_found(self, model_registry):
        with pytest.raises(ValueError, match="not found"):
            model_registry.get_model("nonexistent")

    def test_get_all_models(self, model_registry):
        all_models = model_registry.get_all_models()
        assert isinstance(all_models, dict)
        assert len(all_models) == 4

    def test_add_model(self, model_registry):
        model_registry.add_model(
            model_id="test_model",
            name="test-reranker-v1",
            ipfs_cid="bafytest123",
            metadata={"accuracy": 0.95},
        )
        model = model_registry.get_model("test_model")
        assert model["name"] == "test-reranker-v1"
        assert model["ipfs_cid"] == "bafytest123"
        assert model["accuracy"] == 0.95

    def test_update_model_cid(self, model_registry):
        new_cid = "bafynewcid456"
        model_registry.update_model_cid("minilm", new_cid)
        assert model_registry.get_model("minilm")["ipfs_cid"] == new_cid

    def test_update_model_cid_not_found(self, model_registry):
        with pytest.raises(ValueError, match="not found"):
            model_registry.update_model_cid("ghost", "bafyxxx")

    def test_missing_config_file(self, tmp_path):
        registry = ModelRegistry(config_path=str(tmp_path / "nope.toml"))
        assert registry.models == {}
        assert registry.list_models() == []

    def test_custom_model_has_trained_on(self, model_registry):
        model = model_registry.get_model("custom")
        assert model["trained_on"] == "cerebro-knowledge-base"

    def test_save_and_reload(self, models_toml):
        """Registry should persist changes to disk."""
        reg = ModelRegistry(config_path=models_toml)
        reg.add_model("new", name="new-model", ipfs_cid="bafynew")

        # Reload from same file
        reg2 = ModelRegistry(config_path=models_toml)
        assert "new" in reg2.list_models()
        assert reg2.get_model("new")["name"] == "new-model"
