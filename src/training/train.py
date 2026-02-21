#!/usr/bin/env python3
"""
CEREBRO Model Training Pipeline
Fine-tune cross-encoder on domain-specific data
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import wandb
from google.cloud import storage
import psycopg2
import structlog

log = structlog.get_logger()

# Configuration
CONFIG = {
    'base_model': os.getenv('BASE_MODEL', 'microsoft/deberta-v3-base'),
    'output_dir': '/tmp/cerebro-reranker',
    'batch_size': int(os.getenv('BATCH_SIZE', '16')),
    'epochs': int(os.getenv('EPOCHS', '3')),
    'learning_rate': float(os.getenv('LEARNING_RATE', '2e-5')),
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'max_length': 512,
    'gcs_bucket': os.getenv('GCS_BUCKET', 'cerebro-ml-artifacts'),
    'cerebro_db': os.getenv('CEREBRO_DB', 'postgresql://cerebro@localhost/cerebro'),
}


def export_training_data() -> str:
    """Export labeled query-document pairs from CEREBRO"""
    log.info("Exporting training data from CEREBRO DB...")

    conn = psycopg2.connect(CONFIG['cerebro_db'])
    cur = conn.cursor()

    # Query with implicit feedback signals
    cur.execute("""
        WITH labeled_pairs AS (
            SELECT
                sa.query,
                d.content as document,
                CASE
                    -- Explicit labels (if available)
                    WHEN sa.relevance_score IS NOT NULL THEN sa.relevance_score
                    -- Implicit signals
                    WHEN sa.clicked = TRUE AND sa.dwell_time > 30 THEN 0.9
                    WHEN sa.clicked = TRUE AND sa.dwell_time > 10 THEN 0.7
                    WHEN sa.clicked = TRUE THEN 0.5
                    WHEN sa.shown = TRUE AND sa.clicked = FALSE THEN 0.2
                    ELSE NULL
                END as score
            FROM cerebro.search_analytics sa
            JOIN cerebro.documents d ON sa.document_id = d.id
            WHERE sa.created_at > NOW() - INTERVAL '90 days'
        )
        SELECT query, document, score
        FROM labeled_pairs
        WHERE score IS NOT NULL
        ORDER BY RANDOM()
    """)

    data = []
    for query, document, score in cur.fetchall():
        data.append({
            'query': query,
            'document': document,
            'label': float(score)
        })

    cur.close()
    conn.close()

    log.info(f"Exported {len(data)} training pairs")

    # Save locally
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"/tmp/training_data_{timestamp}.jsonl"

    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    # Upload to GCS
    try:
        client = storage.Client()
        bucket = client.bucket(CONFIG['gcs_bucket'])
        blob = bucket.blob(f"datasets/training_{timestamp}.jsonl")
        blob.upload_from_filename(output_file)

        gcs_path = f"gs://{CONFIG['gcs_bucket']}/datasets/training_{timestamp}.jsonl"
        log.info(f"Uploaded to {gcs_path}")
        return gcs_path

    except Exception as e:
        log.warning(f"GCS upload failed: {e}, using local file")
        return output_file


def prepare_dataset(data_path: str):
    """Load and preprocess dataset"""
    log.info("Loading dataset...")

    # Load from file
    data = []
    with open(data_path.replace('gs://', '/gcs/'), 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Create HF dataset
    dataset = Dataset.from_list(data)

    # Train/val split
    split = dataset.train_test_split(test_size=0.1, seed=42)

    log.info(f"Train: {len(split['train'])}, Val: {len(split['test'])}")

    return split['train'], split['test']


def tokenize_function(examples, tokenizer):
    """Tokenize query-document pairs"""
    return tokenizer(
        examples['query'],
        examples['document'],
        truncation=True,
        padding='max_length',
        max_length=CONFIG['max_length']
    )


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred

    # MSE for regression
    mse = ((predictions - labels) ** 2).mean()

    # Ranking metrics
    from scipy.stats import spearmanr
    correlation, _ = spearmanr(predictions, labels)

    return {
        'mse': mse,
        'spearman': correlation
    }


def train():
    """Main training loop"""
    log.info("Starting training pipeline...")

    # Initialize wandb
    if os.getenv('WANDB_API_KEY'):
        wandb.init(
            project='cerebro-reranker',
            config=CONFIG
        )

    # Export data
    data_path = export_training_data()

    # Load dataset
    train_dataset, eval_dataset = prepare_dataset(data_path)

    # Load tokenizer and model
    log.info(f"Loading model: {CONFIG['base_model']}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['base_model'],
        num_labels=1  # Regression task
    )

    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        warmup_steps=CONFIG['warmup_steps'],
        weight_decay=CONFIG['weight_decay'],
        logging_dir=f"{CONFIG['output_dir']}/logs",
        logging_steps=100,
        evaluation_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model='spearman',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to='wandb' if os.getenv('WANDB_API_KEY') else 'none',
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    log.info("Training started...")
    trainer.train()

    # Evaluate
    log.info("Evaluating...")
    eval_results = trainer.evaluate()
    log.info(f"Evaluation results: {eval_results}")

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{CONFIG['output_dir']}/model_{timestamp}"

    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    log.info(f"Model saved to {output_path}")

    # Upload to GCS
    try:
        client = storage.Client()
        bucket = client.bucket(CONFIG['gcs_bucket'])

        for file_path in Path(output_path).rglob('*'):
            if file_path.is_file():
                blob_name = f"models/reranker-{timestamp}/{file_path.relative_to(output_path)}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))

        log.info(f"Model uploaded to gs://{CONFIG['gcs_bucket']}/models/reranker-{timestamp}/")

    except Exception as e:
        log.error(f"GCS upload failed: {e}")

    return output_path


if __name__ == '__main__':
    train()
