/**
 * SecureLLM-MCP Semantic Cache Middleware
 *
 * Flow with reranker enabled:
 *   candidates = fetch top 50 → cosine similarity → pre-filter (threshold 0.7) →
 *     rerank(query, candidates, top_k=5) → filter by threshold (0.85) → return best
 *
 * Flow without reranker (or on failure):
 *   candidates = fetch top 50 → cosine similarity → filter by threshold (0.85) → return best
 */

import { RerankerClient } from "./reranker-client";
import type {
  CacheCandidate,
  SemanticCacheConfig,
  DEFAULT_RERANKER_CONFIG,
  DEFAULT_SEMANTIC_CACHE_CONFIG,
} from "../types/semantic-cache";

export interface VectorStore {
  search(query: string, topK: number): Promise<CacheCandidate[]>;
}

export interface CacheEntry {
  query: string;
  response: string;
  embedding?: number[];
}

export class SemanticCache {
  private config: SemanticCacheConfig;
  private vectorStore: VectorStore;
  private rerankerClient: RerankerClient | null;

  constructor(vectorStore: VectorStore, config: SemanticCacheConfig) {
    this.config = config;
    this.vectorStore = vectorStore;

    // Initialize reranker client if enabled (also respects RERANKER_ENABLED env)
    const envEnabled = process.env.RERANKER_ENABLED === "true";
    const configEnabled = config.reranker?.enabled ?? false;

    if (envEnabled || configEnabled) {
      const rerankerConfig = {
        endpoint:
          config.reranker?.endpoint ??
          process.env.RERANKER_ENDPOINT ??
          "http://localhost:8001",
        timeout: config.reranker?.timeout ?? 2000,
      };
      this.rerankerClient = new RerankerClient(rerankerConfig);
    } else {
      this.rerankerClient = null;
    }
  }

  /**
   * Look up a semantically similar cached response for the given query.
   * Returns null if no candidate passes the similarity threshold.
   */
  async lookup(query: string): Promise<CacheCandidate | null> {
    // Step 1: Fetch top candidates via cosine similarity from vector store
    const candidates = await this.vectorStore.search(
      query,
      this.config.maxCandidates,
    );

    if (candidates.length === 0) {
      return null;
    }

    // Step 2: If reranker is available, use enhanced flow
    if (this.rerankerClient) {
      return this.lookupWithReranker(query, candidates);
    }

    // Step 3: Fallback — standard cosine similarity flow
    return this.lookupStandard(candidates);
  }

  /**
   * Enhanced flow: pre-filter → rerank → threshold check
   */
  private async lookupWithReranker(
    query: string,
    candidates: CacheCandidate[],
  ): Promise<CacheCandidate | null> {
    const preFilterThreshold = this.config.reranker?.preFilterThreshold ?? 0.7;

    // Pre-filter: keep candidates above lower threshold to reduce reranker load
    const preFiltered = candidates.filter(
      (c) => c.score >= preFilterThreshold,
    );

    if (preFiltered.length === 0) {
      return null;
    }

    // Rerank the pre-filtered candidates
    const documents = preFiltered.map((c) => c.content);
    const reranked = await this.rerankerClient!.rerank(query, documents, 5);

    // If reranker returned fallback results (model === "fallback"),
    // fall through to standard flow
    if (reranked.length > 0 && reranked[0].model === "fallback") {
      return this.lookupStandard(candidates);
    }

    // Find best reranked candidate above the final threshold
    for (const result of reranked) {
      if (result.score >= this.config.similarityThreshold) {
        // Match back to original candidate for metadata
        const original = preFiltered.find(
          (c) => c.content === result.document,
        );
        if (original) {
          return { ...original, score: result.score };
        }
      }
    }

    return null;
  }

  /**
   * Standard flow: cosine similarity threshold check (no reranker)
   */
  private lookupStandard(
    candidates: CacheCandidate[],
  ): CacheCandidate | null {
    const best = candidates[0];
    if (best && best.score >= this.config.similarityThreshold) {
      return best;
    }
    return null;
  }
}
