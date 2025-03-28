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
import os
from concurrent.futures import ThreadPoolExecutor

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
                logging.info(f"Fetched {url} with status code {response.status_code}")
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                attempt += 1
                wait_time = math.pow(2, attempt)  # Exponential backoff
                logging.error(f"Error fetching {url}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        logging.error(f"Failed to fetch {url} after {retries} retries")
        return None

    def is_duplicate(self, snippet, collected_snippets):
        """Check if the snippet is a duplicate based on its hash"""
        snippet_hash = hash(snippet["code"])
        for existing_snippet in collected_snippets:
            if hash(existing_snippet["code"]) == snippet_hash:
                logging.info("Duplicate snippet found")
                return True
        return False

    def extract_from_stackoverflow(self, html, url):
        """Extract code snippets from StackOverflow"""
        soup = BeautifulSoup(html, 'html.parser')
        snippets = []
        for pre in soup.select('pre code'):  # Verify this selector matches the current HTML
            snippets.append({
                "code": pre.get_text(),
                "source": url,
                "tags": [],
                "date": datetime.datetime.now().isoformat(),
                "upvotes": 0
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

    def save_model(self, model_path="crawler_model_checkpoint.pth"):
        """Save the crawling model's state"""
        # Example: Dummy model saving logic (replace with actual model saving code)
        with open(model_path, "w") as f:
            f.write("Dummy model checkpoint")  # Replace with actual model saving
        logging.info(f"Model checkpoint saved to {model_path}")

    def is_time_limit_reached(self):
        """Check if the runtime limit has been exceeded"""
        elapsed_time = datetime.datetime.now() - self.start_time
        return elapsed_time.total_seconds() > self.runtime_limit * 60

    def crawl(self, url, site, max_depth=2, delay=2, snippets_collected=[]):
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

        # Compute reward and update the model
        reward = self.compute_reward(snippets)
        state = {"url": url, "site": site, "depth": max_depth}  # Example state
        action = "crawl"  # Example action
        self.update_model(reward, state, action)

        # Stop if we've already collected the maximum allowed snippets
        if len(snippets_collected) >= self.max_snippets:
            logging.info("Reached the maximum number of snippets")
            return snippets_collected

        soup = BeautifulSoup(html, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        valid_links = [link for link in links if site in link and link.startswith('http')]
        logging.info(f"Found {len(valid_links)} valid links on {url}")

        time.sleep(random.uniform(delay, delay + 1))  # Respect server load

        # Parallel crawl for valid links using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(self.crawl, link, site, max_depth-1, delay, snippets_collected): link for link in valid_links}
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
        """Save collected snippets as CSV (append to the same file)"""
        file_exists = os.path.isfile(filename)  # Check if the file already exists
        with open(filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["code", "source", "tags", "date", "upvotes"])
            if not file_exists:
                writer.writeheader()  # Write header only if the file doesn't exist
            writer.writerows(collected_snippets)
        logging.info(f"Snippets appended to {filename}")

    def update_model(self, reward, state, action):
        """Update the crawling model based on the reward signal"""
        # Example: Dummy model update logic (replace with your actual model training code)
        logging.info(f"Updating model with reward: {reward}, state: {state}, action: {action}")
        
        # Simulate model update (replace this with actual model training logic)
        time.sleep(1)  # Simulate training time

    def compute_reward(self, snippets):
        """Compute a reward based on the quality of extracted snippets"""
        # Example: Reward is the number of valid snippets extracted
        return len(snippets)


# Example usage
sites = {
    "stackoverflow": "https://stackoverflow.com/questions/tagged/python",
    "github_gist": "https://gist.github.com/search?q=python",
    "geeksforgeeks": "https://www.geeksforgeeks.org/python-programming-language/"
}

crawler = CodeSnippetCrawler(sites, max_snippets=50, runtime_limit=30, max_retries=3)  # Limit to 50 snippets, 30 minutes runtime
collected_snippets = {}

for site, url in sites.items():
    # Crawl the site and collect snippets
    collected_snippets[site] = crawler.crawl(url, site, max_depth=2)
    
    # Save progress to the same file for each site
    flattened_snippets = [snippet for site_snippets in collected_snippets.values() for snippet in site_snippets]
    crawler.save_as_csv(flattened_snippets, filename=f"code_snippets_partial_{site}.csv")
    crawler.save_model(model_path=f"crawler_model_checkpoint_{site}.pth")
    logging.info(f"Progress and model saved for {site}")

# Save final snippets
flattened_snippets = [snippet for site_snippets in collected_snippets.values() for snippet in site_snippets]
crawler.save_as_json(collected_snippets, filename="code_snippets_final.json")
crawler.save_as_csv(flattened_snippets, filename="code_snippets_final.csv")
crawler.save_model(model_path="crawler_model_final.pth")
logging.info("Crawling complete. Final snippets and model saved.")