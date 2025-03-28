import requests
from bs4 import BeautifulSoup
import time
import random
import re
import logging
from concurrent.futures import ThreadPoolExecutor
import csv
import json

logging.basicConfig(level=logging.INFO)

class CodeSnippetCrawler:
    def __init__(self, sites, max_snippets=100, time_limit=3600, max_retries=3):
        self.sites = sites
        self.visited_urls = set()
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.max_snippets = max_snippets
        self.start_time = time.time()
        self.time_limit = time_limit
        self.max_retries = max_retries

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

    def save_model(self, model_path="crawler_model_checkpoint.pth"):
        """Save the crawling model's state"""
        # Example: Dummy model saving logic (replace with actual model saving code)
        with open(model_path, "w") as f:
            f.write("Dummy model checkpoint")  # Replace with actual model saving
        logging.info(f"Model checkpoint saved to {model_path}")

    def save_as_csv(self, snippets, filename="code_snippets.csv"):
        """Save snippets to a CSV file"""
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Snippet"])
            for snippet in snippets:
                writer.writerow([snippet])
        logging.info(f"Snippets saved to {filename}")

    def save_as_json(self, snippets, filename="code_snippets.json"):
        """Save snippets to a JSON file"""
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(snippets, f, indent=4)
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

# Example usage
sites = {
    "stackoverflow": "https://stackoverflow.com/questions/tagged/python",
    "github_gist": "https://gist.github.com/search?q=python",
    "geeksforgeeks": "https://www.geeksforgeeks.org/python-programming-language/"
}

crawler = CodeSnippetCrawler(sites, max_snippets=50, time_limit=30, max_retries=3)  # Limit to 50 snippets, 30 minutes runtime
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
