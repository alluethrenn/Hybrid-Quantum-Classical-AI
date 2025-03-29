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
def preprocess_text(text, vocab, tokenizer, max_length=10):
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

    def crawl(self, url, site, max_depth=1, delay=1):
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

        time.sleep(random.uniform(delay, delay + 0))  # Respect server load
        
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
#use beautiful soup to extract text and code from /workspaces/Hybrid-Quantum-Classical-AI/codedata.html
html_file_path = "/workspaces/Hybrid-Quantum-Classical-AI/codedata.html"
with open(html_file_path, 'r', encoding='utf-8') as f:
    html_content = f.read()
    # extract text and code
    # use Beautifulsoup to parse the HTML content

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')
# Extract text
text_elements = soup.find_all('p') # Example: extract all paragraphs
text_data = ' '.join([element.get_text() for element in text_elements])
# Extract code
code_elements = soup.find_all('code') # Example: extract all code blocks
code_data = '\n'.join([element.get_text() for element in code_elements])
import re
cleaned_text = re.sub(r'<[^>]+>', '', text_data) # Remove HTML tags
cleaned_text = cleaned_text.strip() # Remove leading/trailing whitespace
cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # Remove extra spacescleaned_code = re.sub(r'#.*', '', code_data) # Remove comments (for Python)
cleaned_text = cleaned_text.strip()
 # Store in a file
with open('output.txt', 'w', encoding='utf-8') as f:
 f.write("Cleaned Text:\n" + cleaned_text)




# Load the LSTM model
model_path = "/workspaces/Hybrid-Quantum-Classical-AI/lstm_model_final.pth"
lstm_model = load_lstm_model(model_path)

# Load or build the vocabulary from the HTML file
html_file_path = "/workspaces/Hybrid-Quantum-Classical-AI/codedata.html"
try:
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Attempt to extract code snippets from <code> tags
        code_elements = soup.find_all('code')
        snippets = [element.get_text(strip=True) for element in code_elements]
        
        # If no <code> tags are found, try alternative parsing logic
        if not snippets:
            logging.warning("No <code> tags found. Attempting to extract other content.")
            pre_elements = soup.find_all('pre')  # Try extracting <pre> tags as a fallback
            snippets = [element.get_text(strip=True) for element in pre_elements]
        
        if not snippets:
            raise ValueError("No code snippets found in the HTML file.")
        
        # Build vocabulary from the extracted snippets
        vocab = build_vocab(snippets, get_tokenizer("basic_english"))
        vocab_path = "/workspaces/Hybrid-Quantum-Classical-AI/vocab.pkl"
        with open(vocab_path, "wb") as f:
            joblib.dump(vocab, f)
except FileNotFoundError:
    logging.error(f"{html_file_path} not found. Please provide the file.")
    exit(1)
except ValueError as e:
    logging.error(f"Error processing {html_file_path}: {e}")
    exit(1)

# Initialize the crawler
crawler = CodeSnippetCrawler(sites, lstm_model, vocab, get_tokenizer("basic_english"))

# Load previous state if exists
crawler.load_state()

# Crawl the sites and collect snippets
collected_snippets = {}
for site, url in sites.items():
    snippets = crawler.crawl(url, site, max_depth=2)
    predictions = crawler.predict_snippets(snippets)
    collected_snippets[site] = {"snippets": snippets, "predictions": predictions}

# Save snippets, state, and model after crawling
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Snippets with Predictions</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
        h1 { color: #333; }
        .site { margin-bottom: 20px; }
        .snippet { background: #f4f4f4; padding: 10px; border: 1px solid #ddd; margin-bottom: 10px; }
        .prediction { color: #555; font-style: italic; }
    </style>
</head>
<body>
    <h1>Code Snippets with Predictions</h1>
"""

for site, data in collected_snippets.items():
    html_content += f"<div class='site'><h2>{site}</h2>"
    for snippet, prediction in zip(data["snippets"], data["predictions"]):
        html_content += f"""
        <div class='snippet'>
            <pre>{snippet}</pre>
            <p class='prediction'>Prediction: {prediction:.4f}</p>
        </div>
        """
    html_content += "</div>"

html_content += """
</body>
</html>
"""

with open("code_snippets_with_predictions.html", "w") as f:
    f.write(html_content)

crawler.snippets = collected_snippets  # Update the snippets
crawler.save_state()  # Save the visited URLs and state

# Save the model
torch.save(lstm_model, model_path)
print("Crawling complete. Snippets, state, model, and predictions saved.")