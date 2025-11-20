import pandas as pd
import tiktoken
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load data
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

# Initialize LLM
llm = ChatOllama(model="llama3.2:1b", temperature=0.001)

# Initialize transformer
llm_transformer = LLMGraphTransformer(
    llm=llm, node_properties=["description"], relationship_properties=["description"]
)


def process_text(text: str):
    doc = Document(page_content=text)
    return llm_transformer.convert_to_graph_documents([doc])


# Test with small sample
NUM_ARTICLES = 5
graph_documents = []

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(process_text, f"{row['title']} {row['text']}")
        for i, row in news.head(NUM_ARTICLES).iterrows()
    ]

    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing documents"
    ):
        graph_document = future.result()
        graph_documents.extend(graph_document)

# Analyze results
empty_count = 0
for doc in graph_documents:
    if not doc.nodes:
        empty_count += 1

print(f"Processed {len(graph_documents)} documents")
print(f"Empty graphs: {empty_count}")
print(
    f"Success rate: {(len(graph_documents) - empty_count) / len(graph_documents) * 100:.1f}%"
)

# Show sample
if graph_documents:
    sample = graph_documents[0]
    print(f"\nSample document nodes: {len(sample.nodes)}")
    if sample.nodes:
        print(f"First node: {sample.nodes[0]}")
