{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.cerebro-cache;

in {
  options.services.cerebro-cache = {
    enable = mkEnableOption "CEREBRO Cache Layer (Redis + IPFS)";

    redis = {
      maxMemory = mkOption {
        type = types.str;
        default = "4gb";
      };

      port = mkOption {
        type = types.port;
        default = 6379;
      };

      persistence = mkOption {
        type = types.bool;
        default = true;
        description = "Enable RDB snapshots";
      };
    };

    ipfsCache = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Pin cache snapshots to IPFS";
      };

      snapshotInterval = mkOption {
        type = types.str;
        default = "daily";
        description = "Systemd calendar spec for IPFS snapshots";
      };
    };
  };

  config = mkIf cfg.enable {
    # Redis server optimizado para ML inference caching
    services.redis.servers.cerebro-cache = {
      enable = true;
      port = cfg.redis.port;

      settings = {
        maxmemory = cfg.redis.maxMemory;
        maxmemory-policy = "allkeys-lru";

        # Persistence
        save = mkIf cfg.redis.persistence [
          "900 1"    # 15min if 1 key changed
          "300 10"   # 5min if 10 keys changed
          "60 10000" # 1min if 10k keys changed
        ];

        # Performance
        tcp-backlog = 511;
        timeout = 0;
        tcp-keepalive = 300;

        # Snapshots
        stop-writes-on-bgsave-error = false;
        rdbcompression = true;
        rdbchecksum = true;
        dbfilename = "cerebro-cache.rdb";

        # AOF disabled (usamos RDB + IPFS snapshots)
        appendonly = false;

        # Memory optimization
        activedefrag = true;
        jemalloc-bg-thread = true;

        # Limits
        maxclients = 10000;

        # Slow log
        slowlog-log-slower-than = 10000; # 10ms
        slowlog-max-len = 128;
      };
    };

    # Cache warmup service
    systemd.services.cerebro-cache-warmup = {
      description = "Warmup CEREBRO cache with frequent queries";
      after = [ "redis-cerebro-cache.service" ];
      wantedBy = [ "multi-user.target" ];

      script = ''
        set -euo pipefail

        # Wait for Redis
        until ${pkgs.redis}/bin/redis-cli -p ${toString cfg.redis.port} ping > /dev/null 2>&1; do
          echo "Waiting for Redis..."
          sleep 1
        done

        # Load top queries from CEREBRO DB
        ${pkgs.postgresql}/bin/psql -t $CEREBRO_DB -c "
          SELECT DISTINCT query
          FROM cerebro.search_analytics
          WHERE created_at > NOW() - INTERVAL '7 days'
          ORDER BY frequency DESC
          LIMIT 1000
        " | while read -r query; do
          # Pre-compute embeddings
          ${pkgs.curl}/bin/curl -s -X POST http://localhost:8000/v1/embed \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"$query\"}" > /dev/null || true
        done

        echo "Cache warmup complete"
      '';

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        Environment = "CEREBRO_DB=postgresql://cerebro@localhost/cerebro";
      };
    };

    # IPFS snapshot backup
    systemd.services.cerebro-cache-snapshot = mkIf cfg.ipfsCache.enable {
      description = "Snapshot Redis cache to IPFS";

      script = ''
        set -euo pipefail

        SNAPSHOT_DIR="/tmp/cerebro-cache-snapshot-$(date +%s)"
        mkdir -p "$SNAPSHOT_DIR"

        # Trigger BGSAVE
        ${pkgs.redis}/bin/redis-cli -p ${toString cfg.redis.port} BGSAVE

        # Wait for save
        while [ "$(${pkgs.redis}/bin/redis-cli -p ${toString cfg.redis.port} LASTSAVE)" = "$(date +%s)" ]; do
          sleep 1
        done

        # Copy RDB
        cp /var/lib/redis-cerebro-cache/cerebro-cache.rdb "$SNAPSHOT_DIR/"

        # Metadata
        cat > "$SNAPSHOT_DIR/metadata.json" <<EOF
        {
          "timestamp": "$(date -Iseconds)",
          "keys": $(${pkgs.redis}/bin/redis-cli -p ${toString cfg.redis.port} DBSIZE | cut -d: -f2),
          "memory_used": "$(${pkgs.redis}/bin/redis-cli -p ${toString cfg.redis.port} INFO memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')"
        }
        EOF

        # Add to IPFS
        CID=$(${pkgs.kubo}/bin/ipfs add -r -Q "$SNAPSHOT_DIR")

        # Pin
        ${pkgs.kubo}/bin/ipfs pin add "$CID"

        # Store CID
        echo "$CID" > /var/lib/cerebro-cache/latest-snapshot.cid

        # Cleanup
        rm -rf "$SNAPSHOT_DIR"

        echo "Snapshot complete: $CID"
      '';

      serviceConfig = {
        Type = "oneshot";
        User = "redis-cerebro-cache";
        StateDirectory = "cerebro-cache";
      };
    };

    systemd.timers.cerebro-cache-snapshot = mkIf cfg.ipfsCache.enable {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = cfg.ipfsCache.snapshotInterval;
        Persistent = true;
        RandomizedDelaySec = "30m";
      };
    };

    # Restore from IPFS on boot
    systemd.services.cerebro-cache-restore = mkIf cfg.ipfsCache.enable {
      description = "Restore Redis cache from IPFS snapshot";
      before = [ "redis-cerebro-cache.service" ];
      wantedBy = [ "multi-user.target" ];

      script = ''
        set -euo pipefail

        if [ ! -f /var/lib/cerebro-cache/latest-snapshot.cid ]; then
          echo "No snapshot found, skipping restore"
          exit 0
        fi

        CID=$(cat /var/lib/cerebro-cache/latest-snapshot.cid)

        # Fetch from IPFS
        ${pkgs.kubo}/bin/ipfs get "$CID" -o /tmp/cache-restore

        # Restore RDB
        cp /tmp/cache-restore/cerebro-cache.rdb /var/lib/redis-cerebro-cache/
        chown redis-cerebro-cache:redis-cerebro-cache /var/lib/redis-cerebro-cache/cerebro-cache.rdb

        rm -rf /tmp/cache-restore

        echo "Cache restored from $CID"
      '';

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
      };
    };

    # Monitoring
    services.prometheus.scrapeConfigs = [{
      job_name = "redis-cerebro-cache";
      static_configs = [{
        targets = [ "127.0.0.1:${toString cfg.redis.port}" ];
      }];
    }];
  };
}
