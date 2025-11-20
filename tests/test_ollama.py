import pytest
import ollama
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()


def test_ollama_connection():
    """Test if Ollama server is reachable and can list models."""
    # Test connection by listing models (using localhost)
    try:
        client = ollama.Client(host="http://localhost:11434")
        models = client.list()
        print(f"Available models: {[m.model for m in models['models']]}")
        assert "models" in models
        assert isinstance(models["models"], list)
    except Exception as e:
        pytest.fail(f"Failed to connect to Ollama server at localhost: {e}")


def test_llama_model_available():
    """Test if llama3.2:1b model is available."""
    try:
        client = ollama.Client(host="http://localhost:11434")
        models = client.list()
        model_names = [m.model for m in models["models"]]
        assert any("llama3.2:1b" in name for name in model_names), (
            f"llama3.2:1b not found in available models: {model_names}"
        )
    except Exception as e:
        pytest.fail(f"Failed to check models: {e}")


def test_chat_ollama_llama():
    """Test ChatOllama with llama3.2:1b model."""
    # Initialize ChatOllama with llama3.2:1b
    llm = ChatOllama(model="llama3.2:1b", temperature=0.001)

    # Send a simple test message
    response = llm.invoke("Say 'Hello World' in one word.")

    # Check if response is not empty
    assert response is not None
    content = str(response.content)
    assert len(content.strip()) > 0
    print(f"ChatOllama response: {content}")
