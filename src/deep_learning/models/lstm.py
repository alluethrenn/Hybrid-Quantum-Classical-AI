import requests
from bs4 import BeautifulSoup
import time
import random
import re
import logging
import json
import datetime
import math
import csv
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
import string
import argparse
import torch.serialization

logging.basicConfig(level=logging.INFO)

class CodeSnippetCrawler:
    def __init__(self, sites, max_snippets=100, runtime_limit=30, max_retries=3):
        self.sites = sites
        self.visited_urls = set()
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.max_snippets = max_snippets  # Limit to the maximum number of snippets
        self.runtime_limit = runtime_limit  # Limit to the max runtime (in minutes)
        self.max_retries = max_retries  # Number of retries for failed requests
        self.start_time = datetime.datetime.now()  # Track the start time of crawling

    def fetch_page(self, url, retries=3):
        """Fetch a page with retries and exponential backoff"""
        attempt = 0
        while attempt < retries:
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                logging.info(f"Successfully fetched page: {url}")
                return response.text
            except requests.RequestException as e:
                attempt += 1
                wait_time = math.pow(2, attempt)  # Exponential backoff
                logging.error(f"Error fetching {url}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        logging.error(f"Failed to fetch page after {retries} retries: {url}")
        return None

    def extract_from_stackoverflow(self, html, url):
        """Extract code snippets from StackOverflow"""
        soup = BeautifulSoup(html, 'html.parser')
        snippets = []
        for pre in soup.select('pre code'):  # Look for <pre><code> tags
            code_text = pre.get_text().strip()
            if code_text:  # Ensure the snippet is not empty
                snippets.append({
                    "code": code_text,
                    "source": url,
                    "tags": [],  # Add tags if needed
                    "date": datetime.datetime.now().isoformat(),
                    "upvotes": 0  # Add upvotes if available
                })
        logging.info(f"Extracted {len(snippets)} snippets from {url}")
        return snippets

    def extract_from_github_gist(self, html, url):
        """Extract code snippets from GitHub Gist"""
        soup = BeautifulSoup(html, 'html.parser')
        snippets = []
        for pre in soup.select('table.highlight td.content code'):
            snippets.append({"code": pre.get_text(), "source": url, "tags": [], "date": datetime.datetime.now().isoformat(), "upvotes": 0})
        return snippets

    def extract_from_geeksforgeeks(self, html, url):
        """Extract code snippets from GeeksforGeeks"""
        soup = BeautifulSoup(html, 'html.parser')
        snippets = []
        for pre in soup.select('div.code-container pre'):
            snippets.append({"code": pre.get_text(), "source": url, "tags": [], "date": datetime.datetime.now().isoformat(), "upvotes": 0})
        return snippets

    def extract_snippets(self, html, site, url):
        """Extract code snippets with metadata"""
        if site == "stackoverflow":
            snippets = self.extract_from_stackoverflow(html, url)
        elif site == "github_gist":
            snippets = self.extract_from_github_gist(html, url)
        elif site == "geeksforgeeks":
            snippets = self.extract_from_geeksforgeeks(html, url)
        else:
            snippets = []
        logging.info(f"Extracted {len(snippets)} snippets from {url}")
        return snippets

    def is_time_limit_reached(self):
        """Check if the runtime limit has been exceeded"""
        elapsed_time = datetime.datetime.now() - self.start_time
        return elapsed_time.total_seconds() > self.runtime_limit * 60

    def crawl(self, url, site, max_depth=10, delay=2, snippets_collected=[]):
        """Crawl a website and collect code snippets with depth limit and runtime check"""
        if url in self.visited_urls or max_depth <= 0 or self.is_time_limit_reached():
            return snippets_collected

        self.visited_urls.add(url)
        logging.info(f"Crawling: {url}")

        html = self.fetch_page(url)
        if not html:
            return snippets_collected

        snippets = self.extract_snippets(html, site, url)
        snippets_collected.extend([s for s in snippets if not self.is_duplicate(s, snippets_collected)])

        if len(snippets_collected) == 0:
            logging.warning("No snippets were collected from this URL. Continuing to the next URL.")
            return snippets_collected

        # Stop if we've already collected the maximum allowed snippets
        if len(snippets_collected) >= self.max_snippets:
            logging.info("Reached the maximum number of snippets")
            return snippets_collected

        soup = BeautifulSoup(html, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        valid_links = [link for link in links if site in link and link.startswith('http')]

        time.sleep(random.uniform(delay, delay + 1))  # Respect server load

        # Parallel crawl for valid links using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(self.crawl, link, site, max_depth - 1, delay, snippets_collected): link for link in valid_links}
            for future in future_to_url:
                result = future.result()  # Fetch results once they are completed
                snippets_collected.extend(result)

        return snippets_collected

    def save_as_json(self, collected_snippets, filename="code_snippets.json"):
        """Save collected snippets as JSON"""
        with open(filename, "w") as f:
            json.dump(collected_snippets, f, indent=4)
        logging.info(f"Snippets saved as {filename}")

    def save_as_csv(self, collected_snippets, filename="code_snippets.csv"):
        """Save collected snippets as CSV"""
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['code', 'source', 'tags', 'date', 'upvotes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for snippet in collected_snippets:
                writer.writerow(snippet)
        logging.info(f"Snippets saved as {filename}")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device='cpu'):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device  # Add device attribute
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def generate_text(self, start_string, gen_length=1000):
        self.eval()
        chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        idx_to_char = {idx: char for idx, char in enumerate(chars)}
        input_seq = torch.zeros(1, len(start_string), len(chars)).to(self.device)
        for i, char in enumerate(start_string):
            input_seq[0, i, char_to_idx[char]] = 1.0
        hidden = (torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device),
                  torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device))
        generated_text = start_string
        for _ in range(gen_length):
            output, hidden = self.lstm(input_seq, hidden)
            output = self.fc(output[:, -1, :])
            _, predicted_idx = torch.max(output, 1)
            predicted_char = idx_to_char[predicted_idx.item()]
            generated_text += predicted_char
            input_seq = torch.zeros(1, len(generated_text), len(chars)).to(self.device)
            for i, char in enumerate(generated_text[-len(start_string):]):
                input_seq[0, i, char_to_idx[char]] = 1.0
        return generated_text


torch.serialization.add_safe_globals([LSTMModel])


def example_function():
    print('This is an example function.')


class ExampleClass:
    def __init__(self, value):
        self.value = value


def main():
    parser = argparse.ArgumentParser(description='LSTM Model for code completion and web crawling')
    parser.add_argument('--url', type=str, default="https://stackoverflow.com/questions/tagged/python", help='URL to crawl')
    parser.add_argument('--depth', type=int, default=1, help='Depth of web crawling')
    parser.add_argument('--seq_length', type=int, default=1000, help='Length of input sequences')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--max_text_length', type=int, default=5000, help='Maximum length of text data to use')
    parser.add_argument('--model_path', type=str, default='lstm_model.pth', help='Path to save/load the model')
    parser.add_argument('--start_string', type=str, default="start_string", help='Starting string for text generation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation
    model = LSTMModel(input_size=len(chars), hidden_size=128, output_size=len(chars), num_layers=2, device=device).to(device)

    # Use CodeSnippetCrawler to crawl the web and collect text data
    sites = {
        "stackoverflow": "https://stackoverflow.com/questions/tagged/python",
        "github_gist": "https://gist.github.com/search?q=python",
        "geeksforgeeks": "https://www.geeksforgeeks.org/python-programming-language/"
    }
    crawler = CodeSnippetCrawler(sites)
    text_data = crawler.crawl(args.url, site="stackoverflow", max_depth=args.depth)
    text_data = ' '.join([snippet['code'] for snippet in text_data])  # Extract code snippets

    # Limit the length of the crawled text
    max_length = args.max_text_length
    if len(text_data) > max_length:
        text_data = text_data[:max_length]
        logging.info(f"Text data truncated to {max_length} characters.")

    logging.info(f"Length of crawled text data: {len(text_data)}")
    if len(text_data.strip()) == 0:
        logging.warning("No text data was crawled. Loading fallback data from codedata.json.")
        try:
            with open("codedata.json", "r") as f:
                fallback_data = json.load(f)
                if isinstance(fallback_data, list) and all(isinstance(snippet, dict) for snippet in fallback_data):
                    text_data = ' '.join([snippet['code'] for snippet in fallback_data if 'code' in snippet])
                else:
                    logging.error("Fallback data is not in the expected format. Exiting.")
                    return
        except FileNotFoundError:
            logging.error("Fallback file codedata.json not found. Exiting.")
            return

    # Preprocess and train the LSTM model
    data, char_to_idx, idx_to_char = model.create_dataset(text_data, args.seq_length)
    if len(data) == 0:
        logging.warning("No data available for training. Skipping training.")
    else:
        model.train_model(data, char_to_idx, args.num_epochs, args.batch_size, args.learning_rate)

    # Save the trained model
    model.save_model(args.model_path)

    # Load and save the model with fixed weights
    model = torch.load("lstm_model.pth", weights_only=False)
    torch.save(model.state_dict(), "lstm_model_fixed.pth")

    # Generate text using the trained model
    generated_text = model.generate_text(args.start_string, gen_length=1000)
    print(f'Generated text: {generated_text}')


if __name__ == '__main__':
    main()