#!/usr/bin/env python3
"""
Performance benchmarking script
"""

import time
import asyncio
import statistics
from typing import List
import httpx
import structlog
from rich.console import Console
from rich.table import Table
from rich.progress import track

log = structlog.get_logger()
console = Console()


async def benchmark_endpoint(
    url: str,
    payload: dict,
    requests: int = 100,
    concurrency: int = 10
) -> dict:
    """Benchmark a single endpoint"""

    latencies = []
    errors = 0

    async def make_request():
        nonlocal errors
        try:
            async with httpx.AsyncClient() as client:
                start = time.time()
                resp = await client.post(url, json=payload, timeout=30.0)
                latency = (time.time() - start) * 1000

                if resp.status_code == 200:
                    latencies.append(latency)
                else:
                    errors += 1
        except Exception as e:
            errors += 1
            log.error("Request failed", error=str(e))

    # Run concurrent requests
    console.print(f"[yellow]Running {requests} requests with {concurrency} concurrent...[/yellow]")

    for i in track(range(0, requests, concurrency), description="Benchmarking"):
        tasks = [make_request() for _ in range(min(concurrency, requests - i))]
        await asyncio.gather(*tasks)

    return {
        'total_requests': requests,
        'successful': len(latencies),
        'errors': errors,
        'min_ms': min(latencies) if latencies else 0,
        'max_ms': max(latencies) if latencies else 0,
        'mean_ms': statistics.mean(latencies) if latencies else 0,
        'median_ms': statistics.median(latencies) if latencies else 0,
        'p95_ms': statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
        'p99_ms': statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
    }


async def main():
    """Run benchmark suite"""

    console.print("[bold blue]🚀 CEREBRO Reranker Benchmark Suite[/bold blue]\n")

    # Test cases
    test_cases = [
        {
            'name': 'Small Query (10 docs)',
            'payload': {
                'query': 'kubernetes security best practices',
                'documents': [f'Document {i} about kubernetes' for i in range(10)],
                'mode': 'auto'
            }
        },
        {
            'name': 'Medium Query (50 docs)',
            'payload': {
                'query': 'machine learning deployment strategies',
                'documents': [f'Document {i} about ML' for i in range(50)],
                'mode': 'auto'
            }
        },
        {
            'name': 'Large Query (100 docs)',
            'payload': {
                'query': 'distributed systems architecture patterns',
                'documents': [f'Document {i} about distributed systems' for i in range(100)],
                'mode': 'auto'
            }
        },
        {
            'name': 'Fast Mode (50 docs)',
            'payload': {
                'query': 'python async programming',
                'documents': [f'Document {i} about async' for i in range(50)],
                'mode': 'fast'
            }
        },
        {
            'name': 'Accurate Mode (50 docs)',
            'payload': {
                'query': 'neural network optimization',
                'documents': [f'Document {i} about neural nets' for i in range(50)],
                'mode': 'accurate'
            }
        },
    ]

    results = []

    for test_case in test_cases:
        console.print(f"\n[cyan]Testing: {test_case['name']}[/cyan]")
        result = await benchmark_endpoint(
            'http://localhost:8000/v1/rerank',
            test_case['payload'],
            requests=100,
            concurrency=10
        )
        results.append({'name': test_case['name'], **result})

    # Display results
    console.print("\n[bold green]📊 Benchmark Results[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test Case")
    table.add_column("Success", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")

    for result in results:
        table.add_row(
            result['name'],
            str(result['successful']),
            str(result['errors']),
            f"{result['mean_ms']:.2f}",
            f"{result['p95_ms']:.2f}",
            f"{result['p99_ms']:.2f}",
        )

    console.print(table)


if __name__ == '__main__':
    asyncio.run(main())
