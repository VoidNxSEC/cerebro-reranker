"""
CEREBRO Phantom CLI

Usage:
    cerebro rag query "pergunta" --reranker
    cerebro rag query "pergunta" --top-k 10
"""

import asyncio
import os
import sys

import click
import structlog

log = structlog.get_logger()


@click.group()
def cli():
    """CEREBRO Phantom — Knowledge Engine CLI"""
    pass


@cli.group()
def rag():
    """RAG pipeline commands."""
    pass


@rag.command("query")
@click.argument("question")
@click.option(
    "--reranker",
    is_flag=True,
    default=False,
    help="Enable reranker for improved result quality.",
)
@click.option(
    "--top-k",
    default=5,
    type=int,
    show_default=True,
    help="Number of documents to retrieve.",
)
@click.option(
    "--reranker-url",
    default=None,
    help="Reranker service URL (default: http://localhost:8001).",
)
def rag_query(question: str, reranker: bool, top_k: int, reranker_url: str):
    """Execute a RAG query against the knowledge base."""
    asyncio.run(_rag_query(question, reranker, top_k, reranker_url))


async def _rag_query(
    question: str, use_reranker: bool, top_k: int, reranker_url: str | None
):
    """Async implementation of the RAG query command."""
    from phantom.core.rag.engine import RAGEngine
    from phantom.providers.reranker.client import PhantomRerankerClient

    # Determine reranker availability: --reranker flag OR env var
    reranker_enabled = use_reranker or (
        os.getenv("CEREBRO_RERANKER_ENABLED", "").lower() == "true"
    )

    reranker_client = None
    if reranker_enabled:
        endpoint = (
            reranker_url
            or os.getenv("CEREBRO_RERANKER_URL", "http://localhost:8001")
        )
        reranker_client = PhantomRerankerClient(endpoint=endpoint)
        click.echo(f"Reranker enabled: {endpoint}")

    # These would be injected from the actual CEREBRO configuration.
    # For CLI standalone usage, we need concrete implementations.
    try:
        vector_store = _get_vector_store()
        llm_provider = _get_llm_provider()
    except Exception as e:
        click.echo(f"Error initializing providers: {e}", err=True)
        sys.exit(1)

    engine = RAGEngine(
        vector_store=vector_store,
        llm_provider=llm_provider,
        reranker=reranker_client,
    )

    try:
        result = await engine.query(
            query=question, top_k=top_k, use_reranker=reranker_enabled
        )

        click.echo(f"\nAnswer: {result.answer}")
        click.echo(f"\nReranked: {result.reranked}")
        click.echo(f"Latency: {result.latency_ms:.1f}ms")
        click.echo(f"Sources ({len(result.sources)}):")
        for i, source in enumerate(result.sources):
            score = f" (score: {result.scores[i]:.3f})" if result.scores else ""
            click.echo(f"  {i + 1}. {source[:120]}{score}")

    except Exception as e:
        click.echo(f"Query failed: {e}", err=True)
        sys.exit(1)
    finally:
        if reranker_client:
            await reranker_client.close()


def _get_vector_store():
    """Get configured vector store provider. Placeholder for actual CEREBRO integration."""
    raise NotImplementedError(
        "Vector store provider not configured. "
        "Set up CEREBRO providers before using the CLI."
    )


def _get_llm_provider():
    """Get configured LLM provider. Placeholder for actual CEREBRO integration."""
    raise NotImplementedError(
        "LLM provider not configured. "
        "Set up CEREBRO providers before using the CLI."
    )


if __name__ == "__main__":
    cli()
