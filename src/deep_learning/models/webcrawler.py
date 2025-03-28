import requests
from bs4 import BeautifulSoup
import time
import random
import re
import logging
import json

logging.basicConfig(level=logging.INFO)

class CodeSnippetCrawler:
    def __init__(self, sites):
        self.sites = sites
        self.visited_urls = set()
        self.headers = {"User-Agent": "Mozilla/5.0"}

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

# Example usage
sites = {
    "stackoverflow": "https://stackoverflow.com/questions/tagged/python",
    "github_gist": "https://gist.github.com/search?q=python",
    "geeksforgeeks": "https://www.geeksforgeeks.org/python-programming-language/",
    "wikipedia": "https://en.wikipedia.org/wiki/Python_(programming_language)"
}

crawler = CodeSnippetCrawler(sites)
collected_snippets = {}

for site, url in sites.items():
    collected_snippets[site] = crawler.crawl(url, site, max_depth=2)

# Save snippets
with open("code_snippets.json", "w") as f:
    json.dump(collected_snippets, f, indent=4)

print("Crawling complete. Snippets saved.")
