#!/usr/bin/env python3
"""
Graph RAG Testing CLI - Interactive TUI for testing LangChain BAML GraphRAG functionality
"""

import asyncio
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    ListView,
    ListItem,
    Log,
    ProgressBar,
    Static,
    TabbedContent,
    TabPane,
)
from textual import events
from textual.binding import Binding

from tests.test_ollama import (
    test_ollama_connection,
    test_gpt_oss_available,
    test_chat_ollama_gpt_oss,
)


class GraphRAGTester(App):
    """Interactive TUI for testing GraphRAG functionality."""

    CSS = """
    Screen {
        background: $surface;
    }

    Container {
        height: 100%;
        padding: 1;
    }

    #sidebar {
        width: 30;
        background: $panel;
        border-right: solid $primary;
    }

    #main-content {
        width: 100%;
        height: 100%;
    }

    Button {
        margin: 1;
    }

    ProgressBar {
        margin: 1 0;
    }

    DataTable {
        height: 100%;
    }

    Log {
        height: 100%;
        background: $surface;
        border: solid $primary;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "run_all", "Run All Tests"),
        Binding("c", "clear", "Clear Results"),
    ]

    def __init__(self):
        super().__init__()
        self.test_results = {}
        self.current_test = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            with Horizontal():
                with Vertical(id="sidebar"):
                    yield Label("🧪 GraphRAG Tests", classes="title")
                    yield ListView(
                        ListItem(Label("🔗 Ollama Connection")),
                        ListItem(Label("📊 Data Loading")),
                        ListItem(Label("🕸️ Graph Extraction")),
                        id="test-list",
                    )
                    yield Button("Run All Tests", id="run-all", variant="primary")
                    yield Button("Clear Results", id="clear")

                with Vertical(id="main-content"):
                    with TabbedContent():
                        with TabPane("Results", id="results"):
                            yield DataTable(id="results-table")
                        with TabPane("Logs", id="logs"):
                            yield Log(id="test-log", auto_scroll=True)
                        with TabPane("Stats", id="stats"):
                            yield Static(id="stats-display")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the interface."""
        table = self.query_one("#results-table", DataTable)
        table.add_columns("Test", "Status", "Duration", "Details")
        table.zebra_stripes = True

        log = self.query_one("#test-log", Log)
        log.write(
            "GraphRAG Testing CLI initialized. Press 'r' to run all tests or select individual tests.\n"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "run-all":
            self.run_all_tests()
        elif event.button.id == "clear":
            self.clear_results()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle test selection."""
        selected_item = event.item
        # Get the label text directly from the ListItem
        test_name = selected_item.label

        if "Ollama" in test_name:
            self.run_ollama_tests()
        elif "Data" in test_name:
            self.run_data_tests()
        elif "Graph" in test_name:
            self.run_graph_tests()

    async def run_test_async(self, test_func, test_name: str):
        """Run a test asynchronously and update UI."""
        import time

        start_time = time.time()

        log = self.query_one("#test-log", Log)
        log.write(f"🔄 Running {test_name}...\n")

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            duration = time.time() - start_time
            status = "✅ PASS"
            details = "Success"

            log.write(f"✅ {test_name} completed in {duration:.2f}s\n")

        except Exception as e:
            duration = time.time() - start_time
            status = "❌ FAIL"
            details = str(e)
            log.write(f"❌ {test_name} failed: {e}\n")

        # Update results table
        table = self.query_one("#results-table", DataTable)
        table.add_row(test_name, status, f"{duration:.2f}s", details)

        # Update stats
        self.update_stats()

    def run_ollama_tests(self):
        """Run Ollama connectivity tests."""
        asyncio.create_task(
            self.run_test_async(test_ollama_connection, "Ollama Connection")
        )
        asyncio.create_task(
            self.run_test_async(test_gpt_oss_available, "GPT-OSS Available")
        )
        asyncio.create_task(
            self.run_test_async(test_chat_ollama_gpt_oss, "ChatOllama GPT-OSS")
        )

    def run_data_tests(self):
        """Run data loading tests."""
        # Since data loading is synchronous, run in thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self._run_data_test_sync)
            asyncio.create_task(self._handle_data_test_result(future))

    async def _handle_data_test_result(self, future):
        """Handle the result of the data test."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, future.result)
            table = self.query_one("#results-table", DataTable)
            table.add_row("Data Loading", "✅ PASS", "N/A", f"Loaded {result} articles")
            log = self.query_one("#test-log", Log)
            log.write(f"✅ Data loading completed: {result} articles\n")
        except Exception as e:
            table = self.query_one("#results-table", DataTable)
            table.add_row("Data Loading", "❌ FAIL", "N/A", str(e))
            log = self.query_one("#test-log", Log)
            log.write(f"❌ Data loading failed: {e}\n")

    def _run_data_test_sync(self):
        """Run data loading test synchronously."""
        import pandas as pd
        import tiktoken

        news = pd.read_csv(
            "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
        )

        def num_tokens_from_string(string: str, model: str = "gpt-4o") -> int:
            encoding = tiktoken.encoding_for_model(model)
            num_tokens = len(encoding.encode(string))
            return num_tokens

        news["tokens"] = [
            num_tokens_from_string(f"{row['title']} {row['text']}")
            for i, row in news.iterrows()
        ]

        return len(news)

    def run_graph_tests(self):
        """Run graph extraction tests."""
        # This would be more complex, so for now just show it's not implemented
        log = self.query_one("#test-log", Log)
        log.write("🕸️ Graph extraction test requires Ollama server to be running.\n")
        log.write("Please ensure Ollama is accessible at the configured host.\n")

    def run_all_tests(self):
        """Run all test suites."""
        log = self.query_one("#test-log", Log)
        log.write("🚀 Running all GraphRAG tests...\n")

        self.run_ollama_tests()
        self.run_data_tests()
        self.run_graph_tests()

    def clear_results(self):
        """Clear all results and logs."""
        table = self.query_one("#results-table", DataTable)
        table.clear()
        table.add_columns("Test", "Status", "Duration", "Details")

        log = self.query_one("#test-log", Log)
        log.clear()
        log.write("Results cleared. Ready to run tests.\n")

        stats = self.query_one("#stats-display", Static)
        stats.update("No tests run yet.")

    def update_stats(self):
        """Update the stats display."""
        table = self.query_one("#results-table", DataTable)
        rows = table.rows

        total = len(rows)
        passed = sum(1 for row in rows if "✅" in row[1])
        failed = sum(1 for row in rows if "❌" in row[1])

        stats_text = f"""
📊 Test Statistics

Total Tests: {total}
✅ Passed: {passed}
❌ Failed: {failed}
📈 Success Rate: {(passed / total * 100):.1f}% if total > 0 else 0%
        """

        stats = self.query_one("#stats-display", Static)
        stats.update(stats_text.strip())


def main():
    """Main entry point."""
    app = GraphRAGTester()
    app.run()


if __name__ == "__main__":
    main()
