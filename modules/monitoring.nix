{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.cerebro-monitoring;

  # Grafana dashboards como código
  rerankDashboard = pkgs.writeTextFile {
    name = "rerank-dashboard.json";
    text = builtins.toJSON {
      dashboard = {
        title = "CEREBRO Reranker Performance";
        tags = [ "cerebro" "ml" "reranking" ];
        timezone = "browser";
        panels = [
          {
            title = "Requests per Second";
            type = "graph";
            targets = [{
              expr = "rate(rerank_requests_total[5m])";
              legendFormat = "{{model}}";
            }];
          }
          {
            title = "P95 Latency by Model";
            type = "graph";
            targets = [{
              expr = "histogram_quantile(0.95, rate(rerank_duration_seconds_bucket[5m]))";
              legendFormat = "{{model}}";
            }];
          }
          {
            title = "Cache Hit Rate";
            type = "stat";
            targets = [{
              expr = "rate(rerank_cache_hits_total[5m]) / rate(rerank_requests_total[5m])";
            }];
          }
          {
            title = "Model Confidence Distribution";
            type = "heatmap";
            targets = [{
              expr = "rerank_confidence_score";
            }];
          }
          {
            title = "GPU Utilization";
            type = "graph";
            targets = [{
              expr = "nvidia_gpu_duty_cycle";
            }];
          }
          {
            title = "IPFS Pin Status";
            type = "table";
            targets = [{
              expr = "ipfs_cluster_pin_status";
            }];
          }
        ];
      };
    };
  };

  # Alert rules
  alertRules = pkgs.writeTextFile {
    name = "cerebro-alerts.yaml";
    text = ''
      groups:
        - name: cerebro_reranker
          interval: 30s
          rules:
            # High latency
            - alert: RerankHighLatency
              expr: histogram_quantile(0.95, rate(rerank_duration_seconds_bucket[5m])) > 1.0
              for: 5m
              labels:
                severity: warning
              annotations:
                summary: "Reranker P95 latency above 1s"
                description: "Model {{ $labels.model }} latency: {{ $value }}s"

            # Low cache hit rate
            - alert: RerankLowCacheHit
              expr: rate(rerank_cache_hits_total[10m]) / rate(rerank_requests_total[10m]) < 0.3
              for: 10m
              labels:
                severity: info
              annotations:
                summary: "Cache hit rate below 30%"

            # Model error rate
            - alert: RerankHighErrorRate
              expr: rate(rerank_errors_total[5m]) / rate(rerank_requests_total[5m]) > 0.05
              for: 5m
              labels:
                severity: critical
              annotations:
                summary: "Error rate above 5%"

            # GPU OOM
            - alert: GPUMemoryPressure
              expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.9
              for: 2m
              labels:
                severity: warning
              annotations:
                summary: "GPU memory usage above 90%"

            # IPFS cluster unhealthy
            - alert: IPFSClusterDegraded
              expr: ipfs_cluster_peers_total < 2
              for: 5m
              labels:
                severity: warning
              annotations:
                summary: "IPFS cluster has less than 2 peers"

            # Training data drift
            - alert: TrainingDataDrift
              expr: cerebro_new_training_samples > 50000
              for: 1h
              labels:
                severity: info
              annotations:
                summary: "50k+ new training samples available"
                description: "Consider triggering model retraining"
    '';
  };

in {
  options.services.cerebro-monitoring = {
    enable = mkEnableOption "CEREBRO Monitoring Stack";

    prometheus = {
      retention = mkOption {
        type = types.str;
        default = "90d";
      };

      scrapeInterval = mkOption {
        type = types.str;
        default = "15s";
      };
    };

    grafana = {
      port = mkOption {
        type = types.port;
        default = 3000;
      };

      adminPassword = mkOption {
        type = types.str;
        description = "Grafana admin password";
      };
    };

    loki = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable Loki for log aggregation";
      };

      retention = mkOption {
        type = types.str;
        default = "30d";
      };
    };

    alerting = {
      enable = mkOption {
        type = types.bool;
        default = true;
      };

      webhookUrl = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Slack/Discord webhook for alerts";
      };
    };
  };

  config = mkIf cfg.enable {
    # Prometheus
    services.prometheus = {
      enable = true;
      retentionTime = cfg.prometheus.retention;

      globalConfig = {
        scrape_interval = cfg.prometheus.scrapeInterval;
        evaluation_interval = "15s";
      };

      scrapeConfigs = [
        # Reranker API
        {
          job_name = "cerebro-reranker";
          static_configs = [{
            targets = [ "127.0.0.1:8000" ];
          }];
          metrics_path = "/metrics";
        }

        # Redis
        {
          job_name = "redis";
          static_configs = [{
            targets = [ "127.0.0.1:6379" ];
          }];
        }

        # IPFS
        {
          job_name = "ipfs";
          static_configs = [{
            targets = [ "127.0.0.1:5001" ];
          }];
          metrics_path = "/debug/metrics/prometheus";
        }

        # IPFS Cluster
        {
          job_name = "ipfs-cluster";
          static_configs = [{
            targets = [ "127.0.0.1:9094" ];
          }];
        }

        # PostgreSQL (CEREBRO DB)
        {
          job_name = "postgresql";
          static_configs = [{
            targets = [ "127.0.0.1:9187" ];
          }];
        }

        # Node Exporter
        {
          job_name = "node";
          static_configs = [{
            targets = [ "127.0.0.1:9100" ];
          }];
        }

        # NVIDIA GPU
        {
          job_name = "nvidia-gpu";
          static_configs = [{
            targets = [ "127.0.0.1:9835" ];
          }];
        }
      ];

      # Alert rules
      rules = [ alertRules ];

      # Alertmanager
      alertmanagers = mkIf cfg.alerting.enable [{
        static_configs = [{
          targets = [ "127.0.0.1:9093" ];
        }];
      }];
    };

    # Alertmanager
    services.prometheus.alertmanager = mkIf cfg.alerting.enable {
      enable = true;

      configuration = {
        global = {
          resolve_timeout = "5m";
        };

        route = {
          group_by = [ "alertname" "cluster" "service" ];
          group_wait = "10s";
          group_interval = "10s";
          repeat_interval = "12h";
          receiver = "default";
        };

        receivers = [
          {
            name = "default";
            webhook_configs = mkIf (cfg.alerting.webhookUrl != null) [{
              url = cfg.alerting.webhookUrl;
              send_resolved = true;
            }];
          }
        ];
      };
    };

    # Grafana
    services.grafana = {
      enable = true;

      settings = {
        server = {
          http_port = cfg.grafana.port;
          domain = "localhost";
        };

        security = {
          admin_password = cfg.grafana.adminPassword;
        };

        analytics.reporting_enabled = false;
      };

      provision = {
        enable = true;

        datasources.settings.datasources = [
          {
            name = "Prometheus";
            type = "prometheus";
            access = "proxy";
            url = "http://127.0.0.1:9090";
            isDefault = true;
          }
          {
            name = "Loki";
            type = "loki";
            access = "proxy";
            url = "http://127.0.0.1:3100";
          }
        ];

        dashboards.settings.providers = [{
          name = "CEREBRO";
          options.path = pkgs.linkFarm "grafana-dashboards" [
            {
              name = "reranker.json";
              path = rerankDashboard;
            }
          ];
        }];
      };
    };

    # Loki (Log aggregation)
    services.loki = mkIf cfg.loki.enable {
      enable = true;

      configuration = {
        auth_enabled = false;

        server = {
          http_listen_port = 3100;
        };

        ingester = {
          lifecycler = {
            address = "127.0.0.1";
            ring = {
              kvstore.store = "inmemory";
              replication_factor = 1;
            };
          };
          chunk_idle_period = "5m";
          chunk_retain_period = "30s";
        };

        schema_config = {
          configs = [{
            from = "2024-01-01";
            store = "boltdb-shipper";
            object_store = "filesystem";
            schema = "v11";
            index = {
              prefix = "index_";
              period = "24h";
            };
          }];
        };

        storage_config = {
          boltdb_shipper = {
            active_index_directory = "/var/lib/loki/index";
            cache_location = "/var/lib/loki/cache";
            shared_store = "filesystem";
          };

          filesystem = {
            directory = "/var/lib/loki/chunks";
          };
        };

        limits_config = {
          enforce_metric_name = false;
          reject_old_samples = true;
          reject_old_samples_max_age = "168h";
          retention_period = cfg.loki.retention;
        };

        table_manager = {
          retention_deletes_enabled = true;
          retention_period = cfg.loki.retention;
        };
      };
    };

    # Promtail (Log shipper)
    services.promtail = mkIf cfg.loki.enable {
      enable = true;

      configuration = {
        server = {
          http_listen_port = 9080;
          grpc_listen_port = 0;
        };

        positions.filename = "/var/lib/promtail/positions.yaml";

        clients = [{
          url = "http://127.0.0.1:3100/loki/api/v1/push";
        }];

        scrape_configs = [
          # Reranker logs
          {
            job_name = "cerebro-reranker";
            static_configs = [{
              targets = [ "localhost" ];
              labels = {
                job = "cerebro-reranker";
                __path__ = "/var/log/cerebro-reranker/*.log";
              };
            }];
          }

          # Systemd journal
          {
            job_name = "journal";
            journal = {
              max_age = "12h";
              labels = {
                job = "systemd-journal";
              };
            };
            relabel_configs = [{
              source_labels = [ "__journal__systemd_unit" ];
              target_label = "unit";
            }];
          }
        ];
      };
    };

    # Exporters
    services.prometheus.exporters = {
      node = {
        enable = true;
        enabledCollectors = [ "systemd" "processes" ];
      };

      postgres = {
        enable = true;
        dataSourceName = "postgresql:///cerebro?host=/run/postgresql";
      };

      redis = {
        enable = true;
        address = "127.0.0.1:6379";
      };
    };

    # NVIDIA GPU exporter
    systemd.services.nvidia-gpu-exporter = {
      description = "NVIDIA GPU Prometheus Exporter";
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        ExecStart = "${pkgs.prometheus-nvidia-gpu-exporter}/bin/nvidia_gpu_exporter";
        Restart = "always";
      };
    };

    # Custom CEREBRO metrics collector
    systemd.services.cerebro-metrics-collector = {
      description = "CEREBRO Custom Metrics Collector";
      wantedBy = [ "multi-user.target" ];
      after = [ "postgresql.service" ];

      script = ''
        ${pkgs.python311.withPackages(ps: [ps.prometheus-client ps.psycopg2])}/bin/python3 - <<'EOF'
        from prometheus_client import start_http_server, Gauge
        import psycopg2
        import time

        # Metrics
        new_training_samples = Gauge('cerebro_new_training_samples', 'New training samples since last training')
        total_queries = Gauge('cerebro_total_queries', 'Total queries in CEREBRO')
        avg_relevance = Gauge('cerebro_avg_relevance_score', 'Average relevance score')

        def collect_metrics():
            conn = psycopg2.connect("postgresql:///cerebro?host=/run/postgresql")
            cur = conn.cursor()

            # New training samples
            cur.execute("""
                SELECT COUNT(*)
                FROM cerebro.search_analytics
                WHERE relevance_score IS NOT NULL
                  AND created_at > (
                      SELECT COALESCE(MAX(trained_at), '1970-01-01')
                      FROM cerebro.model_versions
                  )
            """)
            new_training_samples.set(cur.fetchone()[0])

            # Total queries
            cur.execute("SELECT COUNT(*) FROM cerebro.search_analytics")
            total_queries.set(cur.fetchone()[0])

            # Avg relevance
            cur.execute("SELECT AVG(relevance_score) FROM cerebro.search_analytics WHERE relevance_score IS NOT NULL")
            avg_relevance.set(cur.fetchone()[0] or 0)

            cur.close()
            conn.close()

        if __name__ == '__main__':
            start_http_server(9999)
            while True:
                try:
                    collect_metrics()
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(60)
        EOF
      '';

      serviceConfig = {
        Restart = "always";
        DynamicUser = true;
      };
    };

    # Prometheus scrape do custom collector
    services.prometheus.scrapeConfigs = [{
      job_name = "cerebro-custom-metrics";
      static_configs = [{
        targets = [ "127.0.0.1:9999" ];
      }];
    }];

    # Log rotation
    services.logrotate = {
      enable = true;
      settings = {
        "/var/log/cerebro-reranker/*.log" = {
          frequency = "daily";
          rotate = 7;
          compress = true;
          delaycompress = true;
          notifempty = true;
          create = "0644 cerebro-reranker cerebro-reranker";
          sharedscripts = true;
        };
      };
    };

    # Firewall
    networking.firewall.allowedTCPPorts = [
      cfg.grafana.port  # Grafana
      9090              # Prometheus
      9093              # Alertmanager
      3100              # Loki
    ];
  };
}
