{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.cerebro-reranker;

in {
  options.services.cerebro-reranker = {
    enable = mkEnableOption "CEREBRO Hybrid Reranker Service";

    models = {
      fast = mkOption {
        type = types.str;
        default = "ms-marco-MiniLM-L-6-v2";
        description = "Fast model for initial scoring";
      };

      accurate = mkOption {
        type = types.str;
        default = "ms-marco-deberta-v3-base";
        description = "Accurate model for uncertain cases";
      };

      custom = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Custom fine-tuned model IPFS CID";
      };
    };

    serving = {
      port = mkOption {
        type = types.port;
        default = 8000;
      };

      workers = mkOption {
        type = types.int;
        default = 4;
        description = "Number of uvicorn workers";
      };

      maxBatchSize = mkOption {
        type = types.int;
        default = 32;
      };

      gpuAcceleration = mkOption {
        type = types.bool;
        default = true;
      };
    };

    cache = {
      enable = mkOption {
        type = types.bool;
        default = true;
      };

      ttl = mkOption {
        type = types.int;
        default = 3600;
        description = "Cache TTL in seconds";
      };

      maxMemory = mkOption {
        type = types.str;
        default = "4gb";
      };
    };

    hybrid = {
      confidenceThreshold = mkOption {
        type = types.float;
        default = 0.8;
        description = "Confidence threshold for fast model";
      };

      cloudFallback = mkOption {
        type = types.bool;
        default = true;
        description = "Enable GCP Vertex AI fallback";
      };
    };

    cerebro = {
      database = mkOption {
        type = types.str;
        default = "postgresql://cerebro@localhost/cerebro";
      };

      exportTrainingData = mkOption {
        type = types.bool;
        default = true;
        description = "Auto-export training pairs from CEREBRO";
      };
    };
  };

  config = mkIf cfg.enable {
    # System dependencies
    systemd.services.cerebro-reranker = {
      description = "CEREBRO Hybrid Reranker API";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" "postgresql.service" "redis.service" "ipfs.service" ];

      environment = {
        PYTHONUNBUFFERED = "1";
        MODEL_FAST = cfg.models.fast;
        MODEL_ACCURATE = cfg.models.accurate;
        MODEL_CUSTOM = mkIf (cfg.models.custom != null) cfg.models.custom;
        CONFIDENCE_THRESHOLD = toString cfg.hybrid.confidenceThreshold;
        IPFS_API = "/ip4/127.0.0.1/tcp/5005";
        REDIS_URL = "redis://localhost:6379/0";
        CEREBRO_DB = cfg.cerebro.database;
        CACHE_TTL = toString cfg.cache.ttl;
        MAX_BATCH_SIZE = toString cfg.serving.maxBatchSize;
        CUDA_VISIBLE_DEVICES = mkIf cfg.serving.gpuAcceleration "0";
      };

      serviceConfig = {
        Type = "exec";
        ExecStart = ''
          ${pkgs.python311.withPackages(ps: with ps; [
            fastapi uvicorn sentence-transformers torch-bin
          ])}/bin/uvicorn server:app \
            --host 0.0.0.0 \
            --port ${toString cfg.serving.port} \
            --workers ${toString cfg.serving.workers}
        '';

        Restart = "always";
        RestartSec = "10s";

        # Security hardening
        DynamicUser = true;
        StateDirectory = "cerebro-reranker";
        CacheDirectory = "cerebro-reranker";

        # GPU access
        DeviceAllow = mkIf cfg.serving.gpuAcceleration [
          "/dev/nvidia0 rwm"
          "/dev/nvidiactl rwm"
          "/dev/nvidia-uvm rwm"
        ];

        # Resource limits
        MemoryMax = "8G";
        CPUQuota = "400%";  # 4 cores

        # Capabilities
        AmbientCapabilities = "";
        CapabilityBoundingSet = "";
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ "/var/lib/cerebro-reranker" "/var/cache/cerebro-reranker" ];
      };
    };

    # Training data export automation
    systemd.services.cerebro-export-training = mkIf cfg.cerebro.exportTrainingData {
      description = "Export training data from CEREBRO to GCS";

      script = ''
        ${pkgs.postgresql}/bin/psql ${cfg.cerebro.database} -c "
          COPY (
            SELECT
              query,
              relevant_docs,
              click_through_rate,
              dwell_time
            FROM cerebro.search_analytics
            WHERE relevance_score IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 100000
          ) TO STDOUT CSV HEADER
        " | ${pkgs.google-cloud-sdk}/bin/gsutil cp - \
          gs://cerebro-training/datasets/training_$(date +%Y%m%d).csv
      '';

      serviceConfig = {
        Type = "oneshot";
        User = "postgres";
      };
    };

    systemd.timers.cerebro-export-training = mkIf cfg.cerebro.exportTrainingData {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = "daily";
        Persistent = true;
        RandomizedDelaySec = "1h";
      };
    };

    # Prometheus metrics
    services.prometheus.scrapeConfigs = [{
      job_name = "cerebro-reranker";
      static_configs = [{
        targets = [ "127.0.0.1:${toString cfg.serving.port}" ];
      }];
      metrics_path = "/metrics";
    }];

    # Nginx reverse proxy
    services.nginx.virtualHosts."rerank.cerebro.local" = {
      locations = {
        "/" = {
          proxyPass = "http://127.0.0.1:${toString cfg.serving.port}";
          extraConfig = ''
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_buffering off;
          '';
        };

        "/metrics" = {
          proxyPass = "http://127.0.0.1:${toString cfg.serving.port}/metrics";
        };

        "/health" = {
          proxyPass = "http://127.0.0.1:${toString cfg.serving.port}/health";
        };
      };
    };
  };
}
