# Progressive migration orchestrator
{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.cerebro.reranker.migration;

in {
  config = {
    # Migration automation
    systemd.services.cerebro-migration-orchestrator = {
      description = "Automated migration progression";

      script = ''
        ${pkgs.python311.withPackages(ps: [ps.psycopg2 ps.prometheus-client])}/bin/python3 - <<'EOF'
        import psycopg2
        import time
        from prometheus_client.parser import text_string_to_metric_families
        import urllib.request

        # Thresholds for auto-progression
        AGREEMENT_THRESHOLD = 0.95
        ERROR_RATE_THRESHOLD = 0.01
        LATENCY_IMPROVEMENT = 0.8  # Local should be 20%+ faster

        conn = psycopg2.connect("${config.services.cerebro.reranker.integration.postgresConnection}")

        def get_metrics():
            """Fetch Prometheus metrics"""
            with urllib.request.urlopen('http://localhost:8002/migration/metrics') as f:
                metrics = {}
                for family in text_string_to_metric_families(f.read().decode('utf-8')):
                    for sample in family.samples:
                        metrics[sample.name] = sample.value
                return metrics

        def check_progression_criteria():
            """Check if we can progress to next stage"""
            cur = conn.cursor()

            # Last 1000 requests
            cur.execute("""
                SELECT
                    AVG(agreement_score) as avg_agreement,
                    AVG(local_latency_ms) / NULLIF(AVG(vertex_latency_ms), 0) as latency_ratio,
                    COUNT(*) FILTER (WHERE served_from = 'local_error') * 1.0 / COUNT(*) as error_rate
                FROM cerebro.reranker_migration_metrics
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC
                LIMIT 1000
            """)

            result = cur.fetchone()
            if not result:
                return False

            avg_agreement, latency_ratio, error_rate = result

            print(f"Agreement: {avg_agreement:.3f} (threshold: {AGREEMENT_THRESHOLD})")
            print(f"Latency ratio: {latency_ratio:.3f} (threshold: {LATENCY_IMPROVEMENT})")
            print(f"Error rate: {error_rate:.4f} (threshold: {ERROR_RATE_THRESHOLD})")

            # All criteria must pass
            return (
                avg_agreement >= AGREEMENT_THRESHOLD and
                latency_ratio <= LATENCY_IMPROVEMENT and
                error_rate <= ERROR_RATE_THRESHOLD
            )

        # Main loop
        current_mode = "${cfg.mode}"
        progression = {
            "shadow": "canary",
            "canary": "full"
        }

        if current_mode in progression:
            if check_progression_criteria():
                next_mode = progression[current_mode]
                print(f"✓ Criteria met! Progressing from {current_mode} to {next_mode}")

                # Update NixOS config
                import subprocess
                subprocess.run([
                    "sudo", "nixos-rebuild", "switch",
                    "--flake", ".#cerebro",
                    "--option", "services.cerebro.reranker.migration.mode", next_mode
                ])
            else:
                print(f"Criteria not met, staying in {current_mode} mode")
        else:
            print(f"Already in final mode: {current_mode}")

        conn.close()
        EOF
      '';

      serviceConfig = {
        Type = "oneshot";
      };
    };

    # Run migration check daily
    systemd.timers.cerebro-migration-orchestrator = {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = "daily";
        Persistent = true;
      };
    };
  };
}
