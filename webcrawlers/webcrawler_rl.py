import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces

class WebCrawlerEnv(gym.Env):
    def __init__(self, sites):
        super(WebCrawlerEnv, self).__init__()

        self.sites = list(sites.values())  # Convert dict to list of URLs
        self.visited_urls = set()
        self.current_url = random.choice(self.sites)
        self.headers = {"User-Agent": "Mozilla/5.0"}
        
        # Action Space: Choose between N URLs (fixed size)
        self.action_space = spaces.Discrete(len(self.sites))
        
        # Observation Space: Encoded representation of the visited URLs
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.sites),), dtype=np.float32)

    def fetch_page(self, url):
        """Fetch the page content."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    def extract_snippets(self, html):
        """Extract code snippets from the page."""
        soup = BeautifulSoup(html, 'html.parser')
        snippets = [pre.get_text() for pre in soup.find_all('pre')]
        return snippets

    def step(self, action):
        """Perform an action (visit a new URL) and return the reward."""
        self.current_url = self.sites[action]

        if self.current_url in self.visited_urls:
            return self._get_observation(), -1, False, {}  # Penalty for revisiting

        self.visited_urls.add(self.current_url)
        html = self.fetch_page(self.current_url)
        
        if not html:
            return self._get_observation(), -1, False, {}  # Penalty for bad page

        snippets = self.extract_snippets(html)
        reward = len(snippets)  # Reward based on snippet count
        
        return self._get_observation(), reward, False, {}

    def _get_observation(self):
        """Return a binary vector representing visited URLs."""
        return np.array([1 if site in self.visited_urls else 0 for site in self.sites], dtype=np.float32)

    def reset(self):
        """Reset the environment at the end of an episode."""
        self.visited_urls = set()
        self.current_url = random.choice(self.sites)
        return self._get_observation()
