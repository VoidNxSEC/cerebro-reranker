#!/usr/bin/env python3
"""
Export trained model to ONNX and quantize
For production deployment
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig
import ipfshttpclient
import structlog
from pathlib import Path
import sys

log = structlog.get_logger()


def export_to_onnx(model_path: str, output_path: str):
    """Export PyTorch model to ONNX"""
    log.info(f"Exporting {model_path} to ONNX...")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Export
    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        model_path,
        export=True
    )

    # Save
    onnx_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    log.info(f"ONNX model saved to {output_path}")


def quantize_model(model_path: str, output_path: str):
    """Quantize ONNX model to INT8"""
    log.info("Quantizing model...")

    # Load ONNX model
    model = ORTModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Quantization config
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=True)

    # Quantize
    quantizer = ORTQuantizer.from_pretrained(model)
    quantizer.quantize(
        quantization_config=qconfig,
        save_directory=output_path
    )

    tokenizer.save_pretrained(output_path)

    log.info(f"Quantized model saved to {output_path}")


def pin_to_ipfs(model_path: str) -> str:
    """Pin model to IPFS and return CID"""
    log.info("Pinning to IPFS...")

    client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')

    # Add directory
    result = client.add(model_path, recursive=True)
    cid = result[-1]['Hash']  # Last item is directory CID

    # Pin
    client.pin.add(cid)

    log.info(f"Model pinned to IPFS: {cid}")

    return cid


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: export.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    # Export to ONNX
    onnx_path = f"{model_path}_onnx"
    export_to_onnx(model_path, onnx_path)

    # Quantize
    quantized_path = f"{model_path}_quantized"
    quantize_model(onnx_path, quantized_path)

    # Pin to IPFS
    cid = pin_to_ipfs(quantized_path)

    print(f"\nModel exported successfully!")
    print(f"IPFS CID: {cid}")
    print(f"\nUpdate models.toml with:")
    print(f'ipfs_cid = "{cid}"')
