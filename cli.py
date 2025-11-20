#!/usr/bin/env python3
"""
Graph RAG Testing CLI - Interactive CLI for testing LangChain BAML GraphRAG functionality
"""

import time
import asyncio
from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

from tests.test_ollama import (
    test_ollama_connection,
    test_gpt_oss_available,
    test_chat_ollama_gpt_oss,
)

app = typer.Typer(
    name="graphrag-test",
    help="Interactive CLI for testing GraphRAG functionality",
    add_completion=False,
)
console = Console()


class TestRunner:
    """Handles test execution and results."""

    def __init__(self):
        self.results = []
        self.stats = {"total": 0, "passed": 0, "failed": 0}

    def run_test(self, test_func, test_name: str, description: str = ""):
        """Run a single test and record results."""
        start_time = time.time()

        with console.status(
            f"[bold green]Running {test_name}...[/bold green]"
        ) as status:
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = asyncio.run(test_func())
                else:
                    result = test_func()

                duration = time.time() - start_time
                status = "✅ PASS"
                details = "Success"

                console.print(
                    f"[green]✅ {test_name} completed in {duration:.2f}s[/green]"
                )

            except Exception as e:
                duration = time.time() - start_time
                status = "❌ FAIL"
                details = str(e)
                console.print(f"[red]❌ {test_name} failed: {e}[/red]")

        self.results.append(
            {
                "name": test_name,
                "status": status,
                "duration": f"{duration:.2f}s",
                "details": details,
            }
        )

        if "PASS" in status:
            self.stats["passed"] += 1
        else:
            self.stats["failed"] += 1
        self.stats["total"] += 1

    def run_data_test(self):
        """Run data loading test."""
        start_time = time.time()

        with console.status(
            "[bold green]Running Data Loading test...[/bold green]"
        ) as status:
            try:
                import pandas as pd

                # Load just a sample to make it faster
                news = pd.read_csv(
                    "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv",
                    nrows=100,  # Load only first 100 rows for speed
                )

                article_count = len(news)
                duration = time.time() - start_time

                console.print(
                    f"[green]✅ Data Loading completed: {article_count} articles loaded in {duration:.2f}s[/green]"
                )

                self.results.append(
                    {
                        "name": "Data Loading",
                        "status": "✅ PASS",
                        "duration": f"{duration:.2f}s",
                        "details": f"Loaded {article_count} articles",
                    }
                )
                self.stats["passed"] += 1

            except Exception as e:
                duration = time.time() - start_time
                console.print(f"[red]❌ Data Loading failed: {e}[/red]")

                self.results.append(
                    {
                        "name": "Data Loading",
                        "status": "❌ FAIL",
                        "duration": f"{duration:.2f}s",
                        "details": str(e),
                    }
                )
                self.stats["failed"] += 1

        self.stats["total"] += 1

    def show_results(self):
        """Display test results in a table."""
        table = Table(title="🧪 GraphRAG Test Results")
        table.add_column("Test", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Details", style="dim")

        for result in self.results:
            status_style = "green" if "PASS" in result["status"] else "red"
            table.add_row(
                result["name"],
                f"[{status_style}]{result['status']}[/{status_style}]",
                result["duration"],
                result["details"],
            )

        console.print(table)

    def show_stats(self):
        """Display test statistics."""
        success_rate = (
            (self.stats["passed"] / self.stats["total"] * 100)
            if self.stats["total"] > 0
            else 0
        )

        stats_text = f"""
📊 Test Statistics

Total Tests: {self.stats["total"]}
✅ Passed: {self.stats["passed"]}
❌ Failed: {self.stats["failed"]}
📈 Success Rate: {success_rate:.1f}%
        """

        panel = Panel.fit(
            stats_text.strip(), title="📈 Statistics", border_style="blue"
        )
        console.print(panel)


@app.callback()
def main():
    """GraphRAG Testing CLI - Interactive testing for LangChain BAML GraphRAG functionality."""
    welcome_text = Text("🧪 GraphRAG Testing CLI", style="bold magenta")
    welcome_text.append(
        "\nInteractive CLI for testing LangChain BAML GraphRAG functionality",
        style="dim",
    )

    panel = Panel.fit(welcome_text, title="🚀 Welcome", border_style="green")
    console.print(panel)


@app.command()
def ollama():
    """Test Ollama server connectivity."""
    runner = TestRunner()

    console.print("[bold blue]🔗 Testing Ollama Connection[/bold blue]")
    runner.run_test(test_ollama_connection, "Ollama Connection")
    runner.run_test(test_gpt_oss_available, "GPT-OSS Available")
    runner.run_test(test_chat_ollama_gpt_oss, "ChatOllama GPT-OSS")

    runner.show_results()
    runner.show_stats()


@app.command()
def data():
    """Test data loading functionality."""
    runner = TestRunner()

    console.print("[bold blue]📊 Testing Data Loading[/bold blue]")
    runner.run_data_test()

    runner.show_results()
    runner.show_stats()


@app.command()
def graph():
    """Test graph extraction functionality."""
    console.print(
        "[bold yellow]🕸️ Graph extraction test requires Ollama server to be running.[/bold yellow]"
    )
    console.print(
        "[dim]Please ensure Ollama is accessible at the configured host before running this test.[/dim]"
    )


@app.command()
def all():
    """Run all tests."""
    runner = TestRunner()

    console.print("[bold green]🚀 Running All GraphRAG Tests[/bold green]")
    console.print()

    # Ollama tests
    console.print("[bold blue]🔗 Testing Ollama Connection[/bold blue]")
    runner.run_test(test_ollama_connection, "Ollama Connection")
    runner.run_test(test_gpt_oss_available, "GPT-OSS Available")
    runner.run_test(test_chat_ollama_gpt_oss, "ChatOllama GPT-OSS")

    # Data tests
    console.print("[bold blue]📊 Testing Data Loading[/bold blue]")
    runner.run_data_test()

    # Graph tests (placeholder)
    console.print(
        "[bold yellow]🕸️ Graph extraction test requires Ollama server to be running.[/bold yellow]"
    )

    console.print()
    runner.show_results()
    runner.show_stats()


@app.command()
def info():
    """Show project information."""
    info_text = """
🎯 GraphRAG Testing CLI

This CLI provides interactive testing for LangChain BAML GraphRAG functionality.

Available commands:
  • ollama  - Test Ollama server connectivity
  • data    - Test data loading functionality
  • graph   - Test graph extraction (requires Ollama)
  • all     - Run all tests
  • info    - Show this information

Usage examples:
  python cli.py ollama
  python cli.py data
  python cli.py all
    """

    panel = Panel.fit(info_text.strip(), title="ℹ️ Information", border_style="cyan")
    console.print(panel)


if __name__ == "__main__":
    app()
