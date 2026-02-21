{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.cerebro-training;

  # Python environment para training
  trainingPython = pkgs.python311.withPackages (ps: with ps; [
    torch-bin
    transformers
    datasets
    accelerate
    optimum
    google-cloud-aiplatform
    google-cloud-storage
    psycopg2
    wandb
    tensorboard
  ]);

in {
  options.services.cerebro-training = {
    enable = mkEnableOption "CEREBRO Model Training Pipeline";

    gcp = {
      project = mkOption {
        type = types.str;
        description = "GCP Project ID";
      };

      region = mkOption {
        type = types.str;
        default = "us-central1";
      };

      bucket = mkOption {
        type = types.str;
        description = "GCS bucket for models/datasets";
        example = "cerebro-ml-artifacts";
      };

      credentialsFile = mkOption {
        type = types.path;
        description = "Path to GCP service account JSON";
      };
    };

    training = {
      baseModel = mkOption {
        type = types.str;
        default = "microsoft/deberta-v3-base";
      };

      batchSize = mkOption {
        type = types.int;
        default = 16;
      };

      epochs = mkOption {
        type = types.int;
        default = 3;
      };

      learningRate = mkOption {
        type = types.float;
        default = 2e-5;
      };
    };

    schedule = {
      autoTrain = mkOption {
        type = types.bool;
        default = true;
        description = "Automatically retrain when new data available";
      };

      interval = mkOption {
        type = types.str;
        default = "weekly";
        description = "Training frequency (systemd calendar)";
      };

      minSamples = mkOption {
        type = types.int;
        default = 10000;
        description = "Minimum new samples before retraining";
      };
    };
  };

  config = mkIf cfg.enable {
    # Training script
    environment.etc."cerebro/train.py".text = ''
      #!/usr/bin/env python3
      import os
      import json
      from pathlib import Path
      from datetime import datetime

      import torch
      from transformers import (
          AutoTokenizer,
          AutoModelForSequenceClassification,
          TrainingArguments,
          Trainer,
      )
      from datasets import load_dataset
      from google.cloud import storage, aiplatform
      import psycopg2

      # Config
      PROJECT_ID = "${cfg.gcp.project}"
      REGION = "${cfg.gcp.region}"
      BUCKET = "${cfg.gcp.bucket}"
      BASE_MODEL = "${cfg.training.baseModel}"

      def export_training_data():
          """Export labeled pairs from CEREBRO DB"""
          conn = psycopg2.connect(os.getenv("CEREBRO_DB"))
          cur = conn.cursor()

          cur.execute("""
              SELECT
                  query,
                  document,
                  CASE
                      WHEN relevance_score > 0.8 THEN 1
                      WHEN relevance_score < 0.3 THEN 0
                      ELSE NULL
                  END as label
              FROM cerebro.search_analytics sa
              JOIN cerebro.documents d ON sa.document_id = d.id
              WHERE relevance_score IS NOT NULL
                AND created_at > NOW() - INTERVAL '90 days'
          """)

          data = []
          for query, doc, label in cur.fetchall():
              if label is not None:
                  data.append({
                      "query": query,
                      "document": doc,
                      "label": label
                  })

          cur.close()
          conn.close()

          # Save locally
          output_file = f"/tmp/training_data_{datetime.now():%Y%m%d}.jsonl"
          with open(output_file, 'w') as f:
              for item in data:
                  f.write(json.dumps(item) + '\n')

          print(f"Exported {len(data)} training samples to {output_file}")

          # Upload to GCS
          client = storage.Client()
          bucket = client.bucket(BUCKET)
          blob = bucket.blob(f"datasets/training_{datetime.now():%Y%m%d}.jsonl")
          blob.upload_from_filename(output_file)

          return f"gs://{BUCKET}/datasets/training_{datetime.now():%Y%m%d}.jsonl"

      def train_model(dataset_path):
          """Fine-tune cross-encoder"""
          # Load data
          dataset = load_dataset('json', data_files=dataset_path.replace('gs://', '/gcs/'))
          dataset = dataset['train'].train_test_split(test_size=0.1)

          # Tokenizer
          tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

          def preprocess(examples):
              return tokenizer(
                  examples['query'],
                  examples['document'],
                  truncation=True,
                  padding='max_length',
                  max_length=512
              )

          train_dataset = dataset['train'].map(preprocess, batched=True)
          eval_dataset = dataset['test'].map(preprocess, batched=True)

          # Model
          model = AutoModelForSequenceClassification.from_pretrained(
              BASE_MODEL,
              num_labels=1  # Regression
          )

          # Training args
          training_args = TrainingArguments(
              output_dir="/tmp/cerebro-reranker",
              num_train_epochs=${toString cfg.training.epochs},
              per_device_train_batch_size=${toString cfg.training.batchSize},
              per_device_eval_batch_size=${toString cfg.training.batchSize},
              learning_rate=${toString cfg.training.learningRate},
              warmup_steps=500,
              weight_decay=0.01,
              logging_dir='/tmp/logs',
              logging_steps=100,
              evaluation_strategy="steps",
              eval_steps=500,
              save_strategy="steps",
              save_steps=1000,
              load_best_model_at_end=True,
              fp16=torch.cuda.is_available(),
              push_to_hub=False,
          )

          # Train
          trainer = Trainer(
              model=model,
              args=training_args,
              train_dataset=train_dataset,
              eval_dataset=eval_dataset,
          )

          trainer.train()

          # Save
          output_dir = f"/tmp/cerebro-reranker-{datetime.now():%Y%m%d}"
          trainer.save_model(output_dir)
          tokenizer.save_pretrained(output_dir)

          # Upload to GCS
          client = storage.Client()
          bucket = client.bucket(BUCKET)
          for file_path in Path(output_dir).rglob('*'):
              if file_path.is_file():
                  blob = bucket.blob(f"models/reranker-{datetime.now():%Y%m%d}/{file_path.relative_to(output_dir)}")
                  blob.upload_from_filename(str(file_path))

          return output_dir

      if __name__ == "__main__":
          # Export data
          dataset_path = export_training_data()

          # Train
          model_path = train_model(dataset_path)

          print(f"Training complete: {model_path}")
    '';

    # Training service (manual trigger)
    systemd.services.cerebro-train-model = {
      description = "Train CEREBRO custom reranker";

      environment = {
        GOOGLE_APPLICATION_CREDENTIALS = cfg.gcp.credentialsFile;
        CEREBRO_DB = "postgresql://cerebro@localhost/cerebro";
        PYTHONUNBUFFERED = "1";
      };

      script = ''
        ${trainingPython}/bin/python /etc/cerebro/train.py
      '';

      serviceConfig = {
        Type = "oneshot";
        User = "cerebro-training";
        DynamicUser = true;
        StateDirectory = "cerebro-training";
      };
    };

    # Auto-training timer
    systemd.timers.cerebro-auto-train = mkIf cfg.schedule.autoTrain {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = cfg.schedule.interval;
        Persistent = true;
      };
    };

    # Check if retraining needed
    systemd.services.cerebro-check-retrain = mkIf cfg.schedule.autoTrain {
      description = "Check if CEREBRO retraining needed";

      script = ''
        NEW_SAMPLES=$(${pkgs.postgresql}/bin/psql -t postgresql://cerebro@localhost/cerebro -c "
          SELECT COUNT(*)
          FROM cerebro.search_analytics
          WHERE relevance_score IS NOT NULL
            AND created_at > (
              SELECT CO
