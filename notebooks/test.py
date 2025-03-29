import requests
from bs4 import BeautifulSoup
import time
import random
import logging
import json
import joblib
import re
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from collections import Counter
from itertools import chain
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)

# Define the LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

# Load the LSTM model
def load_lstm_model(model_path):
    model = torch.load(model_path)  # Load the entire model object
    model.eval()
    return model

# Preprocess text for the LSTM model
def preprocess_text(text, vocab, tokenizer, max_length=100):
    tokens = tokenizer(text)
    indices = [vocab[token] for token in tokens if token in vocab]
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))  # Pad with zeros
    else:
        indices = indices[:max_length]  # Truncate if too long
    return torch.tensor(indices, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Build vocabulary from collected snippets
def build_vocab(snippets, tokenizer):
    tokens = chain.from_iterable(tokenizer(snippet) for snippet in snippets)
    counter = Counter(tokens)
    vocab = {token: idx + 1 for idx, (token, _) in enumerate(counter.most_common())}  # Start indexing from 1
    return vocab

# Web crawler class
class CodeSnippetCrawler:
    def __init__(self, sites, model, vocab, tokenizer):
        self.sites = sites
        self.visited_urls = set()
        self.snippets = {}
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer

    def fetch_page(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

    def extract_snippets(self, html, site):
        soup = BeautifulSoup(html, 'html.parser')
        snippets = []

        if site == "stackoverflow":
            for pre in soup.select('pre code'):
                snippets.append(pre.get_text())
        elif site == "github_gist":
            for pre in soup.select('table.highlight td.content code'):
                snippets.append(pre.get_text())
        elif site == "geeksforgeeks":
            for pre in soup.select('div.code-container pre'): 
                snippets.append(pre.get_text())
        elif site == "wikipedia":
            paragraphs = soup.select('p')
            snippets.extend([p.get_text() for p in paragraphs if len(p.get_text()) > 50])

        return snippets

    def crawl(self, url, site, max_depth=2, delay=2):
        if url in self.visited_urls or max_depth <= 0:
            return []
        
        self.visited_urls.add(url)
        logging.info(f"Crawling: {url}")
        
        html = self.fetch_page(url)
        if not html:
            return []
        
        snippets = self.extract_snippets(html, site)
        
        soup = BeautifulSoup(html, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        valid_links = [link for link in links if site in link and link.startswith('http')]

        time.sleep(random.uniform(delay, delay + 1))  # Respect server load
        
        for link in valid_links:
            snippets.extend(self.crawl(link, site, max_depth - 1, delay))

        return snippets

    def predict_snippets(self, snippets):
        predictions = []
        for snippet in snippets:
            input_tensor = preprocess_text(snippet, self.vocab, self.tokenizer)
            with torch.no_grad():
                prediction = self.model(input_tensor)
                predictions.append(prediction.item())
        return predictions

    def save_state(self, filename="crawler_state.pkl"):
        """Save the crawler's visited URLs and snippets."""
        state = {
            "visited_urls": list(self.visited_urls),
            "snippets": self.snippets
        }
        joblib.dump(state, filename)
        print(f"Crawler state saved to {filename}")

    def load_state(self, filename="crawler_state.pkl"):
        """Load a previously saved crawler state."""
        try:
            state = joblib.load(filename)
            self.visited_urls = set(state["visited_urls"])
            self.snippets = state["snippets"]
            print(f"Loaded crawler state from {filename}")
        except FileNotFoundError:
            print("No previous state found. Starting fresh.")

# Example usage
sites = {
    "stackoverflow": "https://stackoverflow.com/questions/tagged/python",
    "github_gist": "https://gist.github.com/search?q=python",
    "geeksforgeeks": "https://www.geeksforgeeks.org/python-programming-language/",
    "wikipedia": "https://en.wikipedia.org/wiki/Python_(programming_language)"
}

# Load the LSTM model
model_path = "/workspaces/Hybrid-Quantum-Classical-AI/lstm_model_final.pth"
lstm_model = load_lstm_model(model_path)

# Initialize tokenizer and vocabulary
tokenizer = get_tokenizer("basic_english")
print(tokenizer("This is a test"))
dummy_snippets = ["This is a sample snippet for building vocabulary."]
vocab = build_vocab(dummy_snippets, tokenizer)

# Initialize the crawler
crawler = CodeSnippetCrawler(sites, lstm_model, vocab, tokenizer)

# Load previous state if exists
crawler.load_state()

# Crawl the sites and collect snippets
collected_snippets = {}
for site, url in sites.items():
    snippets = crawler.crawl(url, site, max_depth=2)
    predictions = crawler.predict_snippets(snippets)
    collected_snippets[site] = {"snippets": snippets, "predictions": predictions}

# Save snippets and state after crawling
with open("code_snippets_with_predictions.json", "w") as f:
    json.dump(collected_snippets, f, indent=4)

crawler.snippets = collected_snippets  # Update the snippets
crawler.save_state()  # Save the updated state

print("Crawling complete. Snippets and predictions saved.")