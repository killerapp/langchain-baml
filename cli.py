#!/usr/bin/env python3
"""
Graph RAG Testing CLI - Interactive CLI for testing LangChain BAML GraphRAG functionality
"""

import time
import asyncio
import os
import threading
from queue import Queue
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

from langchain_ollama import ChatOllama

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


class StreamingFeedback:
    """Provides visual feedback during BAML streaming operations."""

    def __init__(self, console: Console):
        self.console = console
        self.last_direction = None
        self.token_count = 0
        self.start_time = time.time()
        self.live_display = None

    def start_streaming(self):
        """Initialize streaming feedback display."""
        from rich.live import Live
        from rich.text import Text

        status_text = Text("🔄 Initializing stream...", style="blue")
        self.live_display = Live(
            status_text, console=self.console, refresh_per_second=10
        )
        self.live_display.start()

    def update_feedback(self, direction: str, log=None):
        """Update the streaming feedback display.

        Args:
            direction: "up" for sending tokens, "down" for receiving tokens
            log: FunctionLog object from BAML (optional)
        """
        if not self.live_display:
            return

        from rich.text import Text

        current_time = time.time() - self.start_time
        self.token_count += 1

        if direction == "up":
            arrow = "↑"
            color = "green"
            status = "Sending tokens"
        elif direction == "down":
            arrow = "↓"
            color = "blue"
            status = "Receiving tokens"
        else:
            arrow = "⟳"
            color = "yellow"
            status = "Processing"

        # Create status text with arrow and info
        status_text = Text()
        status_text.append(f"{arrow} ", style=color)
        status_text.append(f"{status}", style="white")
        status_text.append(f" | Tokens: {self.token_count}", style="dim")
        status_text.append(f" | Time: {current_time:.1f}s", style="dim")

        self.live_display.update(status_text)
        self.last_direction = direction

    def finish_streaming(self, success: bool = True):
        """Finalize the streaming feedback display."""
        if not self.live_display:
            return

        from rich.text import Text

        final_time = time.time() - self.start_time
        status_text = Text()

        if success:
            status_text.append("✅ ", style="green")
            status_text.append("Stream completed", style="green")
        else:
            status_text.append("❌ ", style="red")
            status_text.append("Stream failed", style="red")

        status_text.append(f" | Total tokens: {self.token_count}", style="dim")
        status_text.append(f" | Duration: {final_time:.1f}s", style="dim")

        self.live_display.update(status_text)
        self.live_display.stop()

        # Print final newline
        self.console.print()


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
    """Test Ollama server connectivity with concise output."""
    console.print("[bold blue]🔗 Ollama Server Status[/bold blue]")

    # Show configuration
    host = os.getenv("OLLAMA_HOST", "localhost")
    console.print(f"[dim]OLLAMA_HOST:[/dim] {host}")
    console.print(f"[dim]Base URL:[/dim] http://{host}:11434")

    try:
        import ollama

        client = ollama.Client(host=f"http://{host}:11434")

        # Get version info
        try:
            # Try ps command to get version
            import subprocess

            result = subprocess.run(
                ["ollama", "version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                console.print(f"[dim]Version:[/dim] {result.stdout.strip()}")
            else:
                console.print("[dim]Version:[/dim] Unable to detect")
        except:
            console.print("[dim]Version:[/dim] Unable to detect")

        # List models
        models = client.list()
        model_list = [m.model for m in models.get("models", [])]
        if model_list:
            console.print(f"[dim]Available models ({len(model_list)}):[/dim]")
            for model in sorted(model_list):
                marker = " ← current" if "gpt-oss" in model else ""
                console.print(f"  • {model}{marker}")
        else:
            console.print("[dim]Available models:[/dim] None found")

        # Test basic connectivity
        console.print("\n[dim]Testing connectivity...[/dim]")
        llm = ChatOllama(model="llama3.2:1b", temperature=0.001)
        response = llm.invoke("Say 'OK' in one word.")
        if response and hasattr(response, "content") and response.content:
            content = str(response.content).strip()
            console.print("[green]✅ Connection successful[/green]")
            console.print(f"[dim]Test response:[/dim] {content}")
        else:
            console.print("[red]❌ Connection failed - no response[/red]")

    except Exception as e:
        console.print(f"[red]❌ Connection failed: {str(e)[:100]}...[/red]")


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
def compare(
    count: int = typer.Option(2, help="Number of articles to test (default: 2)"),
    stream_feedback: bool = typer.Option(
        False, help="Enable streaming feedback with up/down arrows"
    ),
    timeout: int = typer.Option(
        30, help="Timeout in seconds for each LLM call (default: 30)"
    ),
):
    """Compare standard LangChain vs BAML graph extraction performance."""
    console.print(
        "[bold green]🔬 Comparing LangChain vs BAML Graph Extraction[/bold green]"
    )
    console.print(
        "[yellow]⚠️  This test takes time as it processes real articles with LLM calls[/yellow]"
    )
    console.print()

    # Test data - use just 2 articles for manageable timing
    import pandas as pd

    console.print("[bold blue]📊 Loading test data...[/bold blue]")
    news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv",
        nrows=5,  # Load only 5 rows
    )

    test_articles = [
        f"{row['title']} {row['text']}"[:500] for i, row in news.head(count).iterrows()
    ]  # Use specified count of short articles

    console.print(f"✅ Loaded {len(test_articles)} test articles (shortened for speed)")
    console.print()

    # Test 1: Standard LangChain LLMGraphTransformer
    console.print(
        "[bold blue]🔗 Testing Standard LangChain LLMGraphTransformer[/bold blue]"
    )

    from langchain_ollama import ChatOllama
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_core.documents import Document

    llm = ChatOllama(model="llama3.2:1b", temperature=0.001)
    langchain_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=["description"],
        relationship_properties=["description"],
    )

    langchain_success = 0
    langchain_total = len(test_articles)

    for i, article in enumerate(test_articles):
        console.print(f"  Processing article {i + 1}/{langchain_total}...")
        try:
            # Use threading Timer for more reliable timeout on macOS
            result_queue = Queue()

            def run_langchain():
                try:
                    doc = Document(page_content=article)
                    result = langchain_transformer.convert_to_graph_documents([doc])
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))

            thread = threading.Thread(target=run_langchain)
            thread.start()
            thread.join(timeout=timeout)

            if thread.is_alive():
                console.print(f"  ⏰ Timed out after {timeout}s")
            else:
                status, data = result_queue.get_nowait()
                if status == "success":
                    if data and data[0].nodes:
                        langchain_success += 1
                        console.print(f"  ✅ Success")
                    else:
                        console.print(f"  ❌ No nodes extracted")
                else:
                    console.print(f"  ❌ Failed: {data[:50]}...")

        except Exception as e:
            console.print(f"  ❌ Failed: {str(e)[:50]}...")

    langchain_rate = (langchain_success / langchain_total) * 100
    console.print(
        f"✅ LangChain: {langchain_success}/{langchain_total} articles processed ({langchain_rate:.1f}% success)"
    )
    console.print()

    # Test 2: BAML Implementation
    console.print("[bold blue]🎯 Testing BAML Graph Extraction[/bold blue]")

    try:
        import baml_client as client
        from langchain_core.runnables import chain
        from typing import Any, List
        from langchain_community.graphs.graph_document import (
            GraphDocument as LGGraphDocument,
            Node,
            Relationship,
        )

        # Helper functions from the README
        def _format_nodes(nodes: List[Node]) -> List[Node]:
            return [
                Node(
                    id=el.id.title() if isinstance(el.id, str) else el.id,
                    type=el.type.capitalize() if el.type else "Unknown",
                    properties=el.properties,
                )
                for el in nodes
            ]

        def map_to_base_relationship(rel: Any) -> Relationship:
            source = Node(id=rel.source_node_id, type=rel.source_node_type)
            target = Node(id=rel.target_node_id, type=rel.target_node_type)
            return Relationship(
                source=source, target=target, type=rel.type, properties=rel.properties
            )

        def _format_relationships(rels) -> List[Relationship]:
            relationships = [
                map_to_base_relationship(rel)
                for rel in rels
                if rel.type and rel.source_node_id and rel.target_node_id
            ]
            return [
                Relationship(
                    source=_format_nodes([el.source])[0],
                    target=_format_nodes([el.target])[0],
                    type=el.type.replace(" ", "_").upper(),
                    properties=el.properties,
                )
                for el in relationships
            ]

        @chain
        def get_graph(message):
            graph = client.b.ExtractGraph(graph=message.content)
            return graph

        # Create BAML-powered chain
        from langchain_core.prompts import ChatPromptTemplate

        system_prompt = """
        You are a knowledgeable assistant skilled in extracting entities and their relationships from text.
        Your goal is to create a knowledge graph.
        """

        default_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    (
                        "Tip: Make sure to answer in the correct format and do not include any explanations. "
                        "Use the given format to extract information from the following input: {input}"
                    ),
                ),
            ]
        )

        baml_success = 0
        baml_total = len(test_articles)

        for i, article in enumerate(test_articles):
            console.print(f"  Processing article {i + 1}/{baml_total}...")

            feedback = None
            result = None
            try:
                # Use threading for timeout protection on BAML calls too
                result_queue = Queue()

                def run_baml():
                    try:
                        if stream_feedback:
                            # Create streaming feedback handler
                            nonlocal feedback
                            feedback = StreamingFeedback(console)
                            feedback.start_streaming()

                            def on_tick_callback(event_type: str, log):
                                """Callback for streaming feedback."""
                                if feedback is None:
                                    return
                                # Determine direction based on event type or log content
                                if hasattr(log, "role") and log.role == "user":
                                    feedback.update_feedback("up", log)
                                elif hasattr(log, "role") and log.role in [
                                    "assistant",
                                    "system",
                                ]:
                                    feedback.update_feedback("down", log)
                                else:
                                    # Default to alternating pattern for visual feedback
                                    feedback.update_feedback(
                                        "up" if i % 2 == 0 else "down", log
                                    )

                            # Use streaming BAML call with on_tick callback
                            baml_result = client.b.ExtractGraph(
                                graph=article,
                                baml_options={"on_tick": on_tick_callback},
                            )
                            result_queue.put(("success", baml_result))
                            feedback.finish_streaming(success=True)
                        else:
                            # Use regular non-streaming call
                            baml_result = client.b.ExtractGraph(graph=article)
                            result_queue.put(("success", baml_result))
                    except Exception as e:
                        result_queue.put(("error", str(e)))
                        if feedback:
                            feedback.finish_streaming(success=False)

                thread = threading.Thread(target=run_baml)
                thread.start()
                thread.join(timeout=timeout)

                if thread.is_alive():
                    console.print(f"  ⏰ Timed out after {timeout}s")
                    if feedback:
                        feedback.finish_streaming(success=False)
                else:
                    status, data = result_queue.get_nowait()
                    if status == "success":
                        result = data
                    else:
                        console.print(f"  ❌ Failed: {data[:50]}...")

                if result and hasattr(result, "nodes") and result.nodes:
                    baml_success += 1
                    console.print(f"  ✅ Success")
                else:
                    console.print(f"  ❌ No nodes extracted")

            except Exception as e:
                if feedback:
                    feedback.finish_streaming(success=False)
                console.print(f"  ❌ Failed: {str(e)[:50]}...")

        baml_rate = (baml_success / baml_total) * 100
        console.print(
            f"✅ BAML: {baml_success}/{baml_total} articles processed ({baml_rate:.1f}% success)"
        )
        console.print()

    except ImportError:
        console.print("[red]❌ BAML client not available[/red]")
        baml_rate = 0

    # Results comparison
    console.print("[bold green]📊 Performance Comparison[/bold green]")

    table = Table(title="🧪 LangChain vs BAML Graph Extraction")
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Success Rate", justify="center")
    table.add_column("Improvement", justify="center")

    table.add_row("Standard LangChain", f"{langchain_rate:.1f}%", "-")
    table.add_row(
        "LangChain + BAML",
        f"{baml_rate:.1f}%",
        f"+{baml_rate - langchain_rate:.1f}%"
        if baml_rate > langchain_rate
        else f"{baml_rate - langchain_rate:.1f}%",
    )

    console.print(table)

    # Summary
    if baml_rate > langchain_rate:
        improvement = baml_rate - langchain_rate
        console.print(
            f"[green]🎉 BAML improved success rate by {improvement:.1f}%![/green]"
        )
        console.print(
            "[dim]This demonstrates BAML's fuzzy parsing advantage over strict JSON parsing.[/dim]"
        )
    else:
        console.print(
            "[yellow]⚠️ No significant improvement detected in this small sample.[/yellow]"
        )
        console.print(
            "[dim]Try with more articles or different content for better comparison.[/dim]"
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

This CLI provides interactive testing for LangChain BAML GraphRAG functionality,
including performance comparisons between standard LangChain and BAML-enhanced approaches.

Available commands:
  • ollama  - Test Ollama server connectivity
  • data    - Test data loading functionality
  • graph   - Test graph extraction (requires Ollama)
  • compare - Compare LangChain vs BAML performance
  • all     - Run all tests
  • info    - Show this information

Usage examples:
  uv run cli.py ollama
  uv run cli.py data
  uv run cli.py compare  # 🔥 Key feature: performance comparison
  uv run cli.py all
    """

    panel = Panel.fit(info_text.strip(), title="ℹ️ Information", border_style="cyan")
    console.print(panel)


if __name__ == "__main__":
    app()
