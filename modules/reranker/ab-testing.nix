# A/B testing framework for reranker comparison
{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.cerebro.reranker;

  varianceAnalyzerScript = pkgs.writeText "variance-analyzer.py" ''
    import psycopg2
    import json
    import os
    import numpy as np
    from scipy import stats
    from datetime import datetime

    POSTGRES_URL = os.environ["POSTGRES_URL"]

    conn = psycopg2.connect(POSTGRES_URL)
    cur = conn.cursor()

    # Get recent results
    cur.execute("""
        SELECT
            served_from,
            local_latency_ms,
            vertex_latency_ms,
            agreement_score
        FROM cerebro.reranker_migration_metrics
        WHERE timestamp > NOW() - INTERVAL '24 hours'
          AND mode = 'canary'
    """)

    local_latencies = []
    vertex_latencies = []
    agreements = []

    for row in cur.fetchall():
        if row[0] == 'local' and row[1] is not None:
            local_latencies.append(row[1])
        elif row[2] is not None:
            vertex_latencies.append(row[2])
        if row[3] is not None:
            agreements.append(row[3])

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "sample_size": {"local": len(local_latencies), "vertex": len(vertex_latencies)},
    }

    if local_latencies and vertex_latencies:
        t_stat, p_value = stats.ttest_ind(local_latencies, vertex_latencies)

        report["latency"] = {
            "local_mean_ms": round(np.mean(local_latencies), 2),
            "vertex_mean_ms": round(np.mean(vertex_latencies), 2),
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
        }

        if p_value < 0.05:
            improvement = (1 - np.mean(local_latencies) / np.mean(vertex_latencies)) * 100
            report["latency"]["improvement_pct"] = round(improvement, 1)

    if agreements:
        report["agreement"] = {
            "mean": round(np.mean(agreements), 3),
            "std": round(np.std(agreements), 3),
            "min": round(min(agreements), 3),
            "max": round(max(agreements), 3),
        }

    # Persist results
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cerebro.reranker_ab_reports (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            report JSONB NOT NULL
        )
    """)
    cur.execute(
        "INSERT INTO cerebro.reranker_ab_reports (report) VALUES (%s)",
        (json.dumps(report),)
    )
    conn.commit()

    print(json.dumps(report, indent=2))
    conn.close()
  '';

in {
  config = mkIf (cfg.enable && cfg.migration.mode == "canary") {
    # Variance analysis service
    systemd.services.cerebro-variance-analyzer = {
      description = "Statistical analysis of A/B test results";

      environment = {
        POSTGRES_URL = cfg.integration.postgresConnection;
      };

      serviceConfig = {
        Type = "oneshot";
        ExecStart = "${pkgs.python311.withPackages (ps: with ps; [
          psycopg2 scipy numpy
        ])}/bin/python3 ${varianceAnalyzerScript}";
      };
    };

    systemd.timers.cerebro-variance-analyzer = {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = "hourly";
        Persistent = true;
      };
    };
  };
}
