{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.cerebro-ipfs-cluster;

in {
  options.services.cerebro-ipfs-cluster = {
    enable = mkEnableOption "IPFS Cluster for distributed model serving";

    clusterSecret = mkOption {
      type = types.str;
      description = "Cluster secret (32-byte hex)";
      default = "0024080112209e0da4a5775da5aae0e03bcf0548f567ab10c4093c338edb7f617b968e8718e0c1";
    };

    replicationFactor = mkOption {
      type = types.int;
      default = 2;
      description = "Number of replicas for pinned content";
    };

    modelPins = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "List of model CIDs to pin permanently";
      example = [ "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi" ];
    };
  };

  config = mkIf cfg.enable {
    # IPFS daemon
    services.kubo = {
      enable = true;

      settings = {
        Addresses = {
          API = "/ip4/0.0.0.0/tcp/5001";
          Gateway = "/ip4/0.0.0.0/tcp/8080";
          Swarm = [
            "/ip4/0.0.0.0/tcp/4001"
            "/ip6/::/tcp/4001"
            "/ip4/0.0.0.0/udp/4001/quic"
            "/ip6/::/udp/4001/quic"
          ];
        };

        Datastore.StorageMax = "200GB";

        # Performance tuning
        Swarm = {
          ConnMgr = {
            HighWater = 900;
            LowWater = 600;
            GracePeriod = "20s";
          };
        };

        # Enable experimental features
        Experimental = {
          FilestoreEnabled = true;
          UrlstoreEnabled = true;
          ShardingEnabled = true;
          Libp2pStreamMounting = true;
        };
      };

      dataDir = "/var/lib/ipfs";
      autoMount = true;
    };

    # IPFS Cluster daemon
    systemd.services.ipfs-cluster = {
      description = "IPFS Cluster Daemon";
      after = [ "network.target" "ipfs.service" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        IPFS_CLUSTER_PATH = "/var/lib/ipfs-cluster";
        CLUSTER_SECRET = cfg.clusterSecret;
      };

      preStart = ''
        if [ ! -d /var/lib/ipfs-cluster ]; then
          mkdir -p /var/lib/ipfs-cluster

          # Initialize cluster
          ${pkgs.ipfs-cluster}/bin/ipfs-cluster-service init \
            --consensus crdt

          # Configure
          cat > /var/lib/ipfs-cluster/service.json <<EOF
          {
            "cluster": {
              "secret": "${cfg.clusterSecret}",
              "replication_factor_min": ${toString cfg.replicationFactor},
              "replication_factor_max": ${toString (cfg.replicationFactor + 1)}
            },
            "consensus": {
              "crdt": {
                "cluster_name": "cerebro-models",
                "trusted_peers": ["*"]
              }
            },
            "api": {
              "ipfsproxy": {
                "listen_multiaddress": "/ip4/127.0.0.1/tcp/9095"
              },
              "restapi": {
                "http_listen_multiaddress": "/ip4/0.0.0.0/tcp/9094"
              }
            },
            "ipfs_connector": {
              "ipfshttp": {
                "node_multiaddress": "/ip4/127.0.0.1/tcp/5001",
                "pin_timeout": "2m",
                "unpin_timeout": "3h"
              }
            }
          }
          EOF
        fi
      '';

      serviceConfig = {
        Type = "simple";
        ExecStart = "${pkgs.ipfs-cluster}/bin/ipfs-cluster-service daemon";
        Restart = "always";
        RestartSec = "10s";

        DynamicUser = true;
        StateDirectory = "ipfs-cluster";

        # Security
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ "/var/lib/ipfs-cluster" ];
      };
    };

    # Auto-pin models
    systemd.services.ipfs-pin-models = {
      description = "Pin CEREBRO models to IPFS cluster";
      after = [ "ipfs-cluster.service" ];
      wantedBy = [ "multi-user.target" ];

      script = ''
        set -euo pipefail

        # Wait for cluster to be ready
        until ${pkgs.curl}/bin/curl -s http://localhost:9094/id > /dev/null 2>&1; do
          echo "Waiting for IPFS cluster..."
          sleep 2
        done

        # Pin each model
        ${concatMapStringsSep "\n" (cid: ''
          echo "Pinning ${cid}..."
          ${pkgs.ipfs-cluster}/bin/ipfs-cluster-ctl pin add ${cid} \
            --replication-min ${toString cfg.replicationFactor} \
            --replication-max ${toString (cfg.replicationFactor + 1)} \
            --name "cerebro-model-${cid}" || true
        '') cfg.modelPins}

        echo "Model pinning complete"
      '';

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
      };
    };

    # Monitoring
    services.prometheus.scrapeConfigs = [{
      job_name = "ipfs-cluster";
      static_configs = [{
        targets = [ "127.0.0.1:9094" ];
      }];
      metrics_path = "/metrics";
    }];

    # Firewall
    networking.firewall = {
      allowedTCPPorts = [ 4001 5001 8080 9094 9095 ];
      allowedUDPPorts = [ 4001 ];
    };
  };
}
