import pytest
import ollama
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()


def test_ollama_connection():
    """Test if Ollama server is reachable and can list models."""
    # Check if OLLAMA_HOST is set
    host = os.getenv("OLLAMA_HOST")
    assert host == "192.168.86.20", "OLLAMA_HOST not set correctly"

    # Test connection by listing models
    try:
        client = ollama.Client(host=f"http://{host}:11434")
        models = client.list()
        print(f"Available models: {[m.model for m in models['models']]}")
        assert "models" in models
        assert isinstance(models["models"], list)
    except Exception as e:
        pytest.fail(f"Failed to connect to Ollama server at {host}: {e}")


def test_gpt_oss_available():
    """Test if gpt-oss model is available."""
    try:
        client = ollama.Client(host=f"http://{os.getenv('OLLAMA_HOST')}:11434")
        models = client.list()
        model_names = [m.model for m in models["models"]]
        assert any("gpt-oss" in name for name in model_names), (
            f"gpt-oss not found in available models: {model_names}"
        )
    except Exception as e:
        pytest.fail(f"Failed to check models: {e}")


def test_chat_ollama_gpt_oss():
    """Test ChatOllama with gpt-oss model."""
    # Initialize ChatOllama with gpt-oss
    llm = ChatOllama(model="gpt-oss", temperature=0.001)

    # Send a simple test message
    response = llm.invoke("Say 'Hello World' in one word.")

    # Check if response is not empty
    assert response is not None
    assert len(response.content.strip()) > 0
    print(f"ChatOllama response: {response.content}")
