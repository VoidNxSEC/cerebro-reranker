/**
 * SecureLLM-MCP Reranker HTTP Client
 *
 * Circuit breaker: 3 consecutive failures → bypass for 60s
 * Fallback: returns original order when reranker is unavailable
 */

import type { RerankResponse, RerankResult } from "../types/semantic-cache";

export interface RerankerClientOptions {
  endpoint: string;
  timeout: number;
}

interface CircuitBreakerState {
  failures: number;
  lastFailure: number;
  isOpen: boolean;
}

const CIRCUIT_BREAKER_THRESHOLD = 3;
const CIRCUIT_BREAKER_RESET_MS = 60_000;

export class RerankerClient {
  private endpoint: string;
  private timeout: number;
  private circuit: CircuitBreakerState;

  constructor(options: RerankerClientOptions) {
    this.endpoint = options.endpoint;
    this.timeout = options.timeout;
    this.circuit = { failures: 0, lastFailure: 0, isOpen: false };
  }

  /**
   * Rerank documents using the reranker service.
   * Returns original order on failure (graceful degradation).
   */
  async rerank(
    query: string,
    documents: string[],
    topK: number = 5,
    mode: string = "auto",
  ): Promise<RerankResult[]> {
    if (documents.length === 0) {
      return [];
    }

    // Circuit breaker: if open, check if reset period has elapsed
    if (this.circuit.isOpen) {
      const elapsed = Date.now() - this.circuit.lastFailure;
      if (elapsed < CIRCUIT_BREAKER_RESET_MS) {
        return this.fallback(documents, topK);
      }
      // Half-open: attempt one request
      this.circuit.isOpen = false;
    }

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(`${this.endpoint}/v1/rerank`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, documents, top_k: topK, mode }),
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (!response.ok) {
        throw new Error(`Reranker returned ${response.status}`);
      }

      const data: RerankResponse = await response.json();

      // Reset circuit breaker on success
      this.circuit.failures = 0;
      this.circuit.isOpen = false;

      return data.results;
    } catch {
      this.circuit.failures++;
      this.circuit.lastFailure = Date.now();

      if (this.circuit.failures >= CIRCUIT_BREAKER_THRESHOLD) {
        this.circuit.isOpen = true;
      }

      return this.fallback(documents, topK);
    }
  }

  /**
   * Fallback: return documents in original order with synthetic scores.
   */
  private fallback(documents: string[], topK: number): RerankResult[] {
    return documents.slice(0, topK).map((doc, i) => ({
      document: doc,
      score: 1 - i * (1 / Math.max(documents.length, 1)),
      model: "fallback",
      confidence: 0,
    }));
  }

  /** Expose circuit state for testing/monitoring. */
  get circuitOpen(): boolean {
    return this.circuit.isOpen;
  }

  /** Reset circuit breaker state. */
  resetCircuit(): void {
    this.circuit = { failures: 0, lastFailure: 0, isOpen: false };
  }
}
