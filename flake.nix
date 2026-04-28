{
  description = "CEREBRO Hybrid Reranker - Enterprise-grade semantic search reranking";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ { self, nixpkgs, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" ];

      perSystem = { config, self', inputs', pkgs, system, ... }:
        let
          # Rust toolchain com GPU support
          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [ "rust-src" "rust-analyzer" ];
            targets = [ "x86_64-unknown-linux-gnu" ];
          };

          craneLib = (inputs.crane.mkLib pkgs).overrideToolchain rustToolchain;

           # Rust library com FFI
    libscorer = craneLib.buildPackage {
      src = craneLib.cleanCargoSource ./src/lib;
      pname = "cerebro-scorer";
      version = "0.1.0";

      cargoExtraArgs = "--release --lib";

      buildInputs = with pkgs; [
        openssl
        pkg-config
      ] ++ lib.optionals stdenv.isDarwin [
        darwin.apple_sdk.frameworks.Security
      ];

      # Otimizações máximas
      RUSTFLAGS = "-C target-cpu=native -C opt-level=3 -C lto=fat";

      # CUDA support (override via CUDA_PATH env var at runtime if GPU is available)
      CUDA_PATH = "";

      # Build com AVX2/FMA
      NIX_CFLAGS_COMPILE = "-mavx2 -mfma";
    };

          # Python environment com ML stack
          pythonEnvWithRust = pkgs.python313.withPackages (ps: with ps; [
            fastapi
            uvicorn
            sentence-transformers
            torch-bin
            transformers
            optimum
            onnxruntime
            redis
            ipfshttpclient
            #google-cloud-aiplatform
            psycopg2
            pydantic
            prometheus-client
            structlog
            # Rust FFI binding
            (ps.buildPythonPackage {
              pname = "cerebro-scorer";
              version = "0.1.0";
              src = libscorer;
              format = "setuptools";

              nativeBuildInputs = [ pkgs.maturin ];

              buildPhase = ''
                cp -r ${libscorer}/lib/*.so .
              '';
            })
          ]);


# Model registry - versioned models no IPFS
          modelRegistry = pkgs.writeTextFile {
            name = "models.toml";
            text = ''
              [models.minilm]
              name = "ms-marco-MiniLM-L-6-v2"
              size_mb = 80
              latency_ms = 15
              accuracy = 0.89
              ipfs_cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"

              [models.electra]
              name = "ms-marco-electra-base"
              size_mb = 420
              latency_ms = 45
              accuracy = 0.93
              ipfs_cid = "bafybeie5gq4jxvzmsym6hjlwxej4rwdoxt7wadqvmmwbqi7r27fclha2va"

              [models.deberta]
              name = "ms-marco-deberta-v3-base"
              size_mb = 1400
              latency_ms = 120
              accuracy = 0.96
              ipfs_cid = "bafybeidskjjd4zmr7oh6ku6wp72vvbxyibcli2r6if3ocdcy7jjjusvl2u"

              [models.custom]
              name = "cerebro-security-reranker"
              size_mb = 500
              latency_ms = 60
              accuracy = 0.97
              ipfs_cid = "TBD"  # Updated após training
              trained_on = "cerebro-knowledge-base"
            '';
          };

          # Training container para GCP Vertex AI
          trainingContainer = pkgs.dockerTools.buildLayeredImage {
            name = "cerebro-reranker-trainer";
            tag = "latest";

            contents = with pkgs; [
              pythonEnvWithRust
              cudatoolkit
              cudnn
              git
              libscorer
            ];

            config = {
              Cmd = [ "${pythonEnvWithRust}/bin/python" "/app/train.py" ];
              Env = [
                "PYTHONUNBUFFERED=1"
                "CUDA_VISIBLE_DEVICES=0"
                "TRANSFORMERS_CACHE=/cache/models"
              ];
              WorkingDir = "/app";
            };

            extraCommands = ''
              mkdir -p app cache/models
              cp -r ${./src/training}/* app/
              cp ${modelRegistry} app/models.toml
            '';
          };

          # API Server container
          apiContainer = pkgs.dockerTools.buildLayeredImage {
            name = "cerebro-reranker-api";
            tag = "latest";

            contents = with pkgs; [
              pythonEnvWithRust
              libscorer
              kubo  # IPFS client
              redis
              bash
              coreutils
            ];

            config = {
              Cmd = [
                "${pythonEnvWithRust}/bin/uvicorn"
                "server:app"
                "--host" "0.0.0.0"
                "--port" "8000"
                "--workers" "4"
              ];
              Env = [
                "PYTHONPATH=/app"
                "MODEL_REGISTRY=/app/models.toml"
                "IPFS_API=/ip4/127.0.0.1/tcp/5005"
                "REDIS_URL=redis://localhost:6379"
                "CEREBRO_DB=postgresql://cerebro@localhost/cerebro"
              ];
              ExposedPorts = {
                "8000/tcp" = {};
              };
            };

            extraCommands = ''
              mkdir -p app
              cp -r ${./src/reranker}/* app/
              cp ${libscorer}/lib/libscorer.so app/
              cp ${modelRegistry} app/models.toml
            '';
          };

        in {
          packages = {
            default = self'.packages.reranker-api;

            reranker-api = apiContainer;
            trainer = trainingContainer;
            libscorer = libscorer;

            # CLI tool para management
            cerebro-ctl = pkgs.writeShellScriptBin "cerebro-ctl" ''
              set -euo pipefail

              case "''${1:-}" in
                train)
                  echo "🧠 Starting training pipeline on GCP..."
                  ${pkgs.google-cloud-sdk}/bin/gcloud ai custom-jobs create \
                    --region=us-central1 \
                    --display-name=cerebro-reranker-$(date +%s) \
                    --config=${./configs/gcp-training.yaml}
                  ;;

                deploy)
                  echo "🚀 Deploying reranker stack..."
                  ${pkgs.nixos-rebuild}/bin/nixos-rebuild switch \
                    --flake .#cerebro-reranker
                  ;;

                models)
                  echo "📦 Available models:"
                  ${pkgs.kubo}/bin/ipfs cat $(cat /var/lib/cerebro/model-registry.cid) | \
                    ${pkgs.jq}/bin/jq -r '.models[] | "\(.name) - \(.size_mb)MB - \(.accuracy)"'
                  ;;

                benchmark)
                  echo "⚡ Running benchmarks..."
                  ${pythonEnvWithRust}/bin/python ${./scripts/benchmark.py}
                  ;;

                *)
                  echo "Usage: cerebro-ctl {train|deploy|models|benchmark}"
                  exit 1
                  ;;
              esac
            '';
          };

          # Development shell com tudo
          devShells.default = pkgs.mkShell {
            buildInputs = with pkgs; [
              pythonEnvWithRust
              rustToolchain
              kubo
              redis
              postgresql_16
              google-cloud-sdk
              k9s
              kubectl
              docker
              libscorer
            ];

            shellHook = ''
              export PYTHONPATH="$PWD/src:$PYTHONPATH"
              export LD_LIBRARY_PATH="${pkgs.cudatoolkit}/lib:$LD_LIBRARY_PATH"

              echo "🧠 CEREBRO Reranker Development Environment"
              echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              echo "Python: ${pythonEnvWithRust}/bin/python"
              echo "Rust: ${rustToolchain}/bin/rustc"
              echo "IPFS: http://127.0.0.1:5005"
              echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              echo ""
              echo "Commands:"
              echo "  make dev      - Start local development server"
              echo "  make train    - Fine-tune custom model"
              echo "  make test     - Run test suite"
              echo "  make bench    - Performance benchmarks"
            '';
          };

          # Apps para facilitar
          apps = {
            reranker-api = {
              type = "app";
              program = "${pkgs.writeShellScript "run-api" ''
                ${pythonEnvWithRust}/bin/uvicorn src.reranker.server:app --reload
              ''}";
            };

            train = {
              type = "app";
              program = "${pkgs.writeShellScript "run-training" ''
                ${pythonEnvWithRust}/bin/python src/training/train.py "$@"
              ''}";
            };
          };
        };

      flake = {
        # NixOS modules
        nixosModules = {
          default = self.nixosModules.cerebro-reranker;

          cerebro-reranker = import ./modules/reranker-service.nix;
          ipfs-cluster = import ./modules/ipfs-cluster.nix;
          training-pipeline = import ./modules/training-pipeline.nix;
          cache-layer = import ./modules/cache-layer.nix;
          monitoring = import ./modules/monitoring.nix;
        };

        # NixOS configuration completa
        nixosConfigurations.cerebro-reranker = nixpkgs.lib.nixosSystem {
          system = "x86_64-linux";
          modules = [
            self.nixosModules.cerebro-reranker
            self.nixosModules.ipfs-cluster
            self.nixosModules.cache-layer
            self.nixosModules.monitoring

            ({ config, pkgs, ... }: {
              # Hardware configuration
              boot.kernelModules = [ "nvidia" ];
              hardware.nvidia.modesetting.enable = true;

              # Network
              networking.hostName = "cerebro-reranker";
              networking.firewall.allowedTCPPorts = [ 8000 5001 6379 ];

              # System packages
              environment.systemPackages = with pkgs; [
                self.packages.x86_64-linux.cerebro-ctl
                htop
                nvtop
                tmux
              ];
            })
          ];
        };
      };
    };
}
