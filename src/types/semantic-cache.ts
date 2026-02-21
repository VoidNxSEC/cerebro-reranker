/**
 * SecureLLM-MCP Semantic Cache Types
 */

export interface RerankerConfig {
  enabled: boolean;
  endpoint: string; // default "http://localhost:8001"
  timeout: number; // default 2000ms
  preFilterThreshold: number; // default 0.7
}

export interface SemanticCacheConfig {
  /** Similarity threshold for cache hit (after reranking if enabled) */
  similarityThreshold: number;
  /** Maximum candidates to fetch from vector store */
  maxCandidates: number;
  /** TTL for cache entries in seconds */
  ttlSeconds: number;
  /** Reranker configuration (opt-in) */
  reranker?: RerankerConfig;
}

export interface CacheCandidate {
  id: string;
  content: string;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface RerankResult {
  document: string;
  score: number;
  model: string;
  confidence: number;
}

export interface RerankResponse {
  results: RerankResult[];
  mode_used: string;
  cache_hit: boolean;
  latency_ms: number;
  ipfs_cid?: string;
}

export const DEFAULT_SEMANTIC_CACHE_CONFIG: SemanticCacheConfig = {
  similarityThreshold: 0.85,
  maxCandidates: 50,
  ttlSeconds: 3600,
};

export const DEFAULT_RERANKER_CONFIG: RerankerConfig = {
  enabled: false,
  endpoint: "http://localhost:8001",
  timeout: 2000,
  preFilterThreshold: 0.7,
};
