/**
 * Tests for SecureLLM-MCP RerankerClient
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { RerankerClient } from "../reranker-client";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockSuccessResponse(results: any[]) {
  return {
    ok: true,
    json: async () => ({
      results,
      mode_used: "fast",
      cache_hit: false,
      latency_ms: 10,
    }),
  };
}

function mockErrorResponse(status: number) {
  return { ok: false, status };
}

describe("RerankerClient", () => {
  let client: RerankerClient;

  beforeEach(() => {
    client = new RerankerClient({
      endpoint: "http://localhost:8001",
      timeout: 2000,
    });
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("rerank", () => {
    it("should return reranked results on success", async () => {
      const results = [
        { document: "doc1", score: 0.95, model: "minilm", confidence: 0.9 },
        { document: "doc2", score: 0.8, model: "minilm", confidence: 0.85 },
      ];
      mockFetch.mockResolvedValueOnce(mockSuccessResponse(results));

      const response = await client.rerank("test query", ["doc1", "doc2"], 2);

      expect(response).toHaveLength(2);
      expect(response[0].document).toBe("doc1");
      expect(response[0].score).toBe(0.95);
      expect(mockFetch).toHaveBeenCalledOnce();
    });

    it("should send correct request body", async () => {
      mockFetch.mockResolvedValueOnce(mockSuccessResponse([]));

      await client.rerank("query", ["a", "b"], 5, "fast");

      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8001/v1/rerank");
      expect(JSON.parse(options.body)).toEqual({
        query: "query",
        documents: ["a", "b"],
        top_k: 5,
        mode: "fast",
      });
    });

    it("should return empty array for empty documents", async () => {
      const result = await client.rerank("query", []);
      expect(result).toEqual([]);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("should return fallback on HTTP error", async () => {
      mockFetch.mockResolvedValueOnce(mockErrorResponse(500));

      const result = await client.rerank("query", ["doc1", "doc2"], 2);

      expect(result).toHaveLength(2);
      expect(result[0].model).toBe("fallback");
      expect(result[0].document).toBe("doc1");
    });

    it("should return fallback on network error", async () => {
      mockFetch.mockRejectedValueOnce(new Error("ECONNREFUSED"));

      const result = await client.rerank("query", ["doc1"], 1);

      expect(result).toHaveLength(1);
      expect(result[0].model).toBe("fallback");
    });

    it("should return fallback with decreasing scores", async () => {
      mockFetch.mockRejectedValueOnce(new Error("fail"));

      const result = await client.rerank("q", ["a", "b", "c"], 3);

      expect(result[0].score).toBeGreaterThan(result[1].score);
      expect(result[1].score).toBeGreaterThan(result[2].score);
    });
  });

  describe("circuit breaker", () => {
    it("should not be open initially", () => {
      expect(client.circuitOpen).toBe(false);
    });

    it("should open after 3 consecutive failures", async () => {
      mockFetch.mockRejectedValue(new Error("fail"));

      await client.rerank("q", ["d"]);
      expect(client.circuitOpen).toBe(false);

      await client.rerank("q", ["d"]);
      expect(client.circuitOpen).toBe(false);

      await client.rerank("q", ["d"]);
      expect(client.circuitOpen).toBe(true);
    });

    it("should bypass fetch when circuit is open", async () => {
      // Trip the circuit
      mockFetch.mockRejectedValue(new Error("fail"));
      await client.rerank("q", ["d"]);
      await client.rerank("q", ["d"]);
      await client.rerank("q", ["d"]);

      mockFetch.mockClear();

      // Should not call fetch
      const result = await client.rerank("q", ["doc1"], 1);
      expect(mockFetch).not.toHaveBeenCalled();
      expect(result[0].model).toBe("fallback");
    });

    it("should reset after successful request", async () => {
      // Cause 2 failures (not enough to trip)
      mockFetch.mockRejectedValueOnce(new Error("fail"));
      mockFetch.mockRejectedValueOnce(new Error("fail"));
      await client.rerank("q", ["d"]);
      await client.rerank("q", ["d"]);

      // Succeed
      mockFetch.mockResolvedValueOnce(
        mockSuccessResponse([
          { document: "d", score: 0.9, model: "fast", confidence: 0.8 },
        ]),
      );
      await client.rerank("q", ["d"]);

      expect(client.circuitOpen).toBe(false);
    });

    it("should be resettable", async () => {
      mockFetch.mockRejectedValue(new Error("fail"));
      await client.rerank("q", ["d"]);
      await client.rerank("q", ["d"]);
      await client.rerank("q", ["d"]);

      expect(client.circuitOpen).toBe(true);

      client.resetCircuit();
      expect(client.circuitOpen).toBe(false);
    });
  });
});
