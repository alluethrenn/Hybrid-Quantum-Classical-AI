import os
import requests
from bs4 import BeautifulSoup
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.size(1) == 0:  # Check if sequence length is zero
            raise ValueError("Input sequence length is zero.")
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use the last output in the sequence
        return output

class CodeSnippetCrawler:
    def __init__(self, sites, model_path="lstm_model.pth", max_snippets=100, time_limit=3600, max_retries=3, input_size=100, hidden_size=128, output_size=1):
        self.sites = sites
        self.visited_urls = set()
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.max_snippets = max_snippets
        self.start_time = time.time()
        self.time_limit = time_limit
        self.max_retries = max_retries
        self.model_path = model_path
        self.model = self.load_model(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()  # Adjust based on your task

    def load_model(self, input_size, hidden_size, output_size):
        """Load the LSTM model from the specified path or initialize a new one"""
        model = LSTMModel(input_size, hidden_size, output_size)
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path)
                model.load_state_dict(state_dict, strict=False)
                logging.info(f"Model loaded from {self.model_path}")
            except RuntimeError as e:
                logging.error(f"Error loading model state_dict: {e}")
                logging.warning("Initializing a new model due to state_dict mismatch.")
        else:
            logging.warning(f"Model file not found at {self.model_path}. Using default model.")
        return model

    def create_model_weights(self, input_size, hidden_size, output_size, device="cpu"):
        """Create model weights if they do not exist"""
        if not os.path.exists(self.model_path):
            logging.warning(f"Model file not found at {self.model_path}. Creating new model weights.")
            self.model = LSTMModel(input_size, hidden_size, output_size).to(device)
            torch.save(self.model.state_dict(), self.model_path)
            logging.info(f"New model weights created and saved to {self.model_path}.")
        else:
            logging.info(f"Model weights already exist at {self.model_path}. Loading existing weights.")
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.to(device)
        return self.model

    def train_model(self, snippets):
        """Train the LSTM model with new code snippets"""
        if not snippets:
            logging.warning("No snippets provided for training.")
            return

        inputs, targets = self.prepare_data(snippets)
        if inputs.size(1) == 0:  # Check if sequence length is zero
            logging.error("Input sequence length is zero. Skipping training.")
            return

        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        
        logging.info(f"Model trained with {len(snippets)} new snippets. Loss: {loss.item()}")

    def prepare_data(self, snippets):
        """Prepare the data for model training"""
        inputs = torch.randn(len(snippets), 10, 100)  # Example random tensor for input
        targets = torch.randn(len(snippets), 1)  # Example random tensor for target
        return inputs, targets

    def fetch_page(self, url):
        retries = 0
        while retries < self.max_retries:
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                retries += 1
                logging.error(f"Error fetching {url}: {e}. Retry {retries}/{self.max_retries}")
        return None

    def extract_snippets(self, html, site, url=None):
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

        return snippets

    def is_duplicate(self, snippet, snippets_collected):
        return snippet in snippets_collected

    def is_time_limit_reached(self):
        return time.time() - self.start_time > self.time_limit

    def compute_reward(self, snippets):
        # Placeholder for reward computation logic
        return len(snippets)

    def update_model(self, reward, state, action):
        # Placeholder for model update logic
        logging.info(f"Updating model with reward: {reward}, state: {state}, action: {action}")

    def save_model(self, model_path="lstm_model.pth"):
        """Save the model's state dictionary"""
        torch.save(self.model.state_dict(), model_path)
        logging.info(f"Model state dictionary saved to {model_path}")

    def save_as_csv(self, snippets, filename="datasets/raw/code_snippets.csv"):
        """Save snippets to a CSV file"""
        file_exists = os.path.exists(filename)
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Snippet"])  # Write header only if the file is new
            for snippet in snippets:
                writer.writerow([snippet])
        logging.info(f"Snippets saved to {filename}")

    def save_as_json(self, snippets, filename="datasets/raw/code_snippets.json"):
        """Save snippets to a JSON file"""
        if os.path.exists(filename):
            with open(filename, "r", encoding='utf-8') as f:
                existing_snippets = json.load(f)
        else:
            existing_snippets = []

        existing_snippets.extend(snippets)

        with open(filename, "w", encoding='utf-8') as f:
            json.dump(existing_snippets, f, indent=4)
        logging.info(f"Snippets saved to {filename}")

    def crawl(self, url, site, max_depth=15, delay=2, snippets_collected=[]):
        """Crawl a website, collect code snippets, and train the model"""
        if url in self.visited_urls or max_depth <= 0 or self.is_time_limit_reached():
            logging.info(f"Skipping {url} (already visited or depth limit reached)")
            return snippets_collected
        
        self.visited_urls.add(url)
        logging.info(f"Crawling: {url}")
        
        html = self.fetch_page(url)
        if not html:
            logging.error(f"Failed to fetch HTML for {url}")
            return snippets_collected
        
        snippets = self.extract_snippets(html, site, url)
        snippets_collected.extend([s for s in snippets if not self.is_duplicate(s, snippets_collected)])
        logging.info(f"Collected {len(snippets_collected)} snippets so far")

        # Train the model with new snippets
        self.train_model(snippets)

        # Compute reward and update the model
        reward = self.compute_reward(snippets)
        state = {"url": url, "site": site, "depth": max_depth}
        action = "crawl"
        self.update_model(reward, state, action)

        if len(snippets_collected) >= self.max_snippets:
            logging.info("Reached the maximum number of snippets")
            return snippets_collected

        soup = BeautifulSoup(html, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        valid_links = [link for link in links if site in link and link.startswith('http')]
        logging.info(f"Found {len(valid_links)} valid links on {url}")

        time.sleep(random.uniform(delay, delay + 1))

        with ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(self.crawl, link, site, max_depth-1, delay, snippets_collected): link for link in valid_links}
            for future in future_to_url:
                result = future.result()
                snippets_collected.extend(result)
        
        return snippets_collected

# Example usage
sites = {
    "stackoverflow": "https://stackoverflow.com/questions/tagged/python",
    "github_gist": "https://gist.github.com/search?q=python",
    "geeksforgeeks": "https://www.geeksforgeeks.org/python-programming-language/"
}

crawler = CodeSnippetCrawler(sites, model_path="lstm_model.pth", max_snippets=50, time_limit=30, max_retries=3)
collected_snippets = {}

for site, url in sites.items():
    collected_snippets[site] = crawler.crawl(url, site, max_depth=2)
    flattened_snippets = [snippet for site_snippets in collected_snippets.values() for snippet in site_snippets]
    crawler.save_as_csv(flattened_snippets, filename="datasets/raw/code_snippets.csv")
    crawler.save_model(model_path="lstm_model.pth")

flattened_snippets = [snippet for site_snippets in collected_snippets.values() for snippet in site_snippets]
crawler.save_as_json(flattened_snippets, filename="datasets/raw/code_snippets.json")
crawler.save_model(model_path="lstm_model_final.pth")
logging.info("Crawling complete. Final snippets and model saved.")
