# CEREBRO Reranker Integration Module
# Connects hybrid reranker to existing CEREBRO RAG pipeline

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.cerebro.reranker;

in {
  imports = [
    ./ab-testing.nix
  ];

  options.services.cerebro.reranker = {
    enable = mkEnableOption "CEREBRO Hybrid Reranker";

    migration = {
      mode = mkOption {
        type = types.enum [ "vertex-only" "shadow" "canary" "full" ];
        default = "shadow";
        description = ''
          Migration strategy:
          - vertex-only: Use only Google Vertex AI (baseline)
          - shadow: Run both, log comparisons, serve Vertex results
          - canary: percentage-based traffic to local
          - full: 100% local reranking
        '';
      };

      canaryPercentage = mkOption {
        type = types.int;
        default = 10;
        description = "Traffic percentage for canary mode";
      };

      comparisonLogging = mkOption {
        type = types.bool;
        default = true;
        description = "Log comparison metrics between local and Vertex";
      };
    };

    fallback = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable automatic fallback to Vertex on errors";
      };

      latencyThreshold = mkOption {
        type = types.int;
        default = 2000;
        description = "Latency threshold (ms) to trigger fallback";
      };

      errorThreshold = mkOption {
        type = types.int;
        default = 3;
        description = "Consecutive errors before fallback";
      };
    };

    featureFlags = {
      useSIMD = mkOption {
        type = types.bool;
        default = true;
        description = "Enable Rust SIMD acceleration";
      };

      useIPFSCache = mkOption {
        type = types.bool;
        default = true;
        description = "Enable IPFS-backed caching";
      };

      autoRetrain = mkOption {
        type = types.bool;
        default = false;
        description = "Auto-retrain on data drift";
      };
    };

    integration = {
      vectorStore = mkOption {
        type = types.str;
        default = "pgvector";
        description = "Vector store backend (pgvector, qdrant, weaviate)";
      };

      embeddingModel = mkOption {
        type = types.str;
        default = "BAAI/bge-large-en-v1.5";
        description = "Embedding model for CEREBRO";
      };

      postgresConnection = mkOption {
        type = types.str;
        default = "postgresql://cerebro@localhost/cerebro";
        description = "PostgreSQL connection string";
      };

      rerankerPort = mkOption {
        type = types.port;
        default = 8001;
        description = "Port for the local reranker service";
      };

      integrationPort = mkOption {
        type = types.port;
        default = 8002;
        description = "Port for the integration layer";
      };
    };
  };

  config = mkIf cfg.enable {
    # Enable reranker service
    services.cerebro-reranker = {
      enable = true;

      models = {
        fast = "ms-marco-MiniLM-L-6-v2";
        accurate = "ms-marco-deberta-v3-base";
      };

      serving = {
        port = cfg.integration.rerankerPort;
        workers = 4;
        gpuAcceleration = cfg.featureFlags.useSIMD;
      };

      cache = {
        enable = cfg.featureFlags.useIPFSCache;
        ttl = 3600;
        maxMemory = "4gb";
      };

      cerebro = {
        database = cfg.integration.postgresConnection;
        exportTrainingData = cfg.featureFlags.autoRetrain;
      };
    };

    # Integration service — bridges CEREBRO and reranker
    systemd.services.cerebro-reranker-integration = {
      description = "CEREBRO Reranker Integration Layer";
      wantedBy = [ "multi-user.target" ];
      after = [ "cerebro-reranker.service" "postgresql.service" ];

      environment = {
        MIGRATION_MODE = cfg.migration.mode;
        CANARY_PERCENTAGE = toString cfg.migration.canaryPercentage;
        FALLBACK_ENABLED = if cfg.fallback.enable then "true" else "false";
        LATENCY_THRESHOLD = toString cfg.fallback.latencyThreshold;
        ERROR_THRESHOLD = toString cfg.fallback.errorThreshold;
        POSTGRES_URL = cfg.integration.postgresConnection;
        LOCAL_RERANKER_URL = "http://127.0.0.1:${toString cfg.integration.rerankerPort}";
        LISTEN_PORT = toString cfg.integration.integrationPort;
      };

      serviceConfig = {
        Type = "simple";
        ExecStart = "${pkgs.python311.withPackages(ps: with ps; [
          fastapi uvicorn httpx psycopg2 google-cloud-aiplatform
          structlog prometheus-client xxhash
        ])}/bin/uvicorn cerebro.reranker_client:app --host 127.0.0.1 --port ${toString cfg.integration.integrationPort}";
        Restart = "always";
        RestartSec = "5s";

        # Security
        DynamicUser = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;

        # Resource limits
        MemoryMax = "2G";
        CPUQuota = "200%";
      };
    };

    # Migration state tracking table
    systemd.services.cerebro-migration-tracker = mkIf cfg.migration.comparisonLogging {
      description = "Create reranker migration tracking tables";
      after = [ "postgresql.service" ];
      wantedBy = [ "multi-user.target" ];

      script = ''
        ${pkgs.postgresql}/bin/psql "${cfg.integration.postgresConnection}" <<SQL
          CREATE TABLE IF NOT EXISTS cerebro.reranker_migration_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            mode VARCHAR(20),
            query_id UUID,
            local_latency_ms FLOAT,
            vertex_latency_ms FLOAT,
            local_top_result TEXT,
            vertex_top_result TEXT,
            agreement_score FLOAT,
            served_from VARCHAR(20)
          );

          CREATE INDEX IF NOT EXISTS idx_migration_timestamp
            ON cerebro.reranker_migration_metrics(timestamp);
          CREATE INDEX IF NOT EXISTS idx_migration_mode
            ON cerebro.reranker_migration_metrics(mode);
        SQL
      '';

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
      };
    };

    # Nginx routing
    services.nginx.virtualHosts."rerank.cerebro.local" = {
      locations."/v1/rerank" = {
        proxyPass = "http://127.0.0.1:${toString cfg.integration.integrationPort}";
        extraConfig = ''
          proxy_set_header X-Migration-Mode "${cfg.migration.mode}";
          proxy_read_timeout ${toString (cfg.fallback.latencyThreshold / 1000 + 1)}s;
          proxy_connect_timeout 5s;
        '';
      };

      locations."/metrics" = {
        proxyPass = "http://127.0.0.1:${toString cfg.integration.integrationPort}/metrics";
      };

      locations."/health" = {
        proxyPass = "http://127.0.0.1:${toString cfg.integration.integrationPort}/health";
      };
    };

    # Prometheus scrape
    services.prometheus.scrapeConfigs = [{
      job_name = "cerebro-reranker-integration";
      static_configs = [{
        targets = [ "127.0.0.1:${toString cfg.integration.integrationPort}" ];
      }];
      metrics_path = "/metrics";
    }];
  };
}
