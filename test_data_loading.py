import pandas as pd
import tiktoken

# Load the news articles dataset
news = pd.read_csv(
    "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
)


# Define token counting function
def num_tokens_from_string(string: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Add tokens column
news["tokens"] = [
    num_tokens_from_string(f"{row['title']} {row['text']}")
    for i, row in news.iterrows()
]

print(f"Loaded {len(news)} articles")
print(f"Sample tokens: {news['tokens'].head().tolist()}")
print("Data loading test passed!")
