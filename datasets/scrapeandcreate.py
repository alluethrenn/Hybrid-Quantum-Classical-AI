 import os
import json
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
GITHUB_API_URL = "https://api.github.com/repos/{owner}/{repo}/contents/{path}"
GITHUB_BASE_URL = "https://github.com/{owner}/{repo}/tree/{branch}/"
HEADERS = {
    'Accept': 'application/vnd.github.v3+json'
}

# Step 1: Scrape code snippets from a GitHub repository
def scrape_github_repo(owner, repo, file_extension="py"):
    # Create a list to store the scraped code snippets
    code_snippets = []

    # Fetch the list of files in the repository (root directory and subdirectories)
    url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    response = requests.get(url, headers=HEADERS)
    files = response.json()

    # Check if the repository exists and if the files list is returned
    if response.status_code != 200 or not files:
        print(f"Failed to retrieve files from {repo}.")
        return []

    # Loop through files and find Python files (.py)
    for file in files:
        # If the file is a Python file, fetch its contents
        if file['name'].endswith(file_extension):
            file_url = file['download_url']
            code_response = requests.get(file_url)

            if code_response.status_code == 200:
                # Add the code snippet to the list
                code_snippets.append(code_response.text)
            else:
                print(f"Failed to retrieve file: {file['name']}")

        # If it's a directory, recurse into it
        elif file['type'] == 'dir':
            code_snippets.extend(scrape_github_repo(owner, repo, file_extension))

    return code_snippets

# Step 2: Process code snippets into tokenized datasets
def process_code_snippets(code_snippets):
    # Tokenize the code snippets
    tokenizer = Tokenizer(char_level=False, split=" ")  # Tokenize by words
    tokenizer.fit_on_texts(code_snippets)
    
    sequences = tokenizer.texts_to_sequences(code_snippets)
    padded_sequences = pad_sequences(sequences, padding='post')  # Padding for uniform length
    
    # Convert to dataset
    dataset = {
        "snippets": code_snippets,
        "tokenized_sequences": padded_sequences.tolist(),
        "word_index": tokenizer.word_index
    }
    
    return dataset

# Step 3: Save the processed dataset to a JSON file
def save_processed_dataset(dataset, output_path="processed_dataset.json"):
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Processed dataset saved to {output_path}")

# Step 4: Combine all functions for scraping, processing, and saving the data
def scrape_and_process_repo(owner, repo, output_path="processed_dataset.json"):
    print(f"Scraping GitHub repository {repo}...")
    
    # Scrape code snippets from the GitHub repository
    code_snippets = scrape_github_repo(owner, repo)
    
    if not code_snippets:
        print("No code snippets found.")
        return
    
    print(f"Found {len(code_snippets)} code snippets. Processing...")

    # Process the code snippets into tokenized sequences
    dataset = process_code_snippets(code_snippets)
    
    # Save the processed dataset to a JSON file
    save_processed_dataset(dataset, output_path)

# Example usage: Scraping a GitHub repository and processing the code
scrape_and_process_repo("tensorflow", "tensorflow", output_path="tensorflow_dataset.json")
