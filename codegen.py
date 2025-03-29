import random
import datetime
import json
import csv

# Define templates for different complexity levels
beginner_code = [
    "x = 10\nprint(x)",
    "name = 'Alice'\nprint('Hello, ' + name)",
    "a = 5\nb = 10\nprint(a + b)"
]

starter_code = [
    "for i in range({}):\n    print(i)",
    "if {} > 5:\n    print('Greater than 5!')",
    "while {}:\n    print('Looping!')"
]

intermediate_code = [
    "def add(a, b):\n    return a + b",
    "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n - 1)",
    "try:\n    x = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero!')"
]

advanced_code = [
    "class Person:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n    def greet(self):\n        print(f'Hello, my name is {self.name} and I am {self.age} years old.')",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
    "import threading\n\ndef worker():\n    print('Worker thread started!')\n\nthread = threading.Thread(target=worker)\nthread.start()"
]

pro_code = [
    "def memoize(f):\n    cache = {}\n    def wrapper(*args):\n        if args in cache:\n            return cache[args]\n        result = f(*args)\n        cache[args] = result\n        return result\n    return wrapper\n\n@memoize\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "from functools import reduce\n\ndata = [1, 2, 3, 4, 5]\nresult = reduce(lambda x, y: x * y, data)\nprint(f'Product: {result}')",
    "import asyncio\nasync def say_hello():\n    print('Hello!')\n\nasyncio.run(say_hello())"
]

master_code = [
    "class NeuralNetwork:\n    def __init__(self, layers):\n        self.layers = layers\n        self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)]\n\n    def feedforward(self, X):\n        for layer in self.weights:\n            X = np.dot(X, layer)\n        return X",
    "import torch\nimport torch.nn as nn\nclass SimpleNN(nn.Module):\n    def __init__(self):\n        super(SimpleNN, self).__init__()\n        self.layer1 = nn.Linear(10, 20)\n        self.layer2 = nn.Linear(20, 10)\n    def forward(self, x):\n        x = self.layer1(x)\n        x = self.layer2(x)\n        return x",
    "def train_model(model, data, labels, epochs=10, lr=0.01):\n    for epoch in range(epochs):\n        model.train()\n        optimizer = torch.optim.SGD(model.parameters(), lr=lr)\n        optimizer.zero_grad()\n        predictions = model(data)\n        loss = nn.CrossEntropyLoss()(predictions, labels)\n        loss.backward()\n        optimizer.step()"
]

# Function to generate a code snippet with varying complexity and timestamp
def generate_code_snippet(complexity="starter", language="Python"):
    code_snippet = ""
    label = ""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if language == "Python":
        if complexity == "beginner":
            code_snippet = random.choice(beginner_code)
            label = "Beginner"
        
        elif complexity == "starter":
            code_snippet = random.choice(starter_code).format(random.randint(1, 10))  # Simple loops, if-statements
            label = "Starter"
        
        elif complexity == "intermediate":
            code_snippet = random.choice(intermediate_code)
            label = "Intermediate"
        
        elif complexity == "advanced":
            code_snippet = random.choice(advanced_code)
            label = "Advanced"
        
        elif complexity == "pro":
            code_snippet = random.choice(pro_code)
            label = "Pro"
        
        elif complexity == "master":
            code_snippet = random.choice(master_code)
            label = "Master"
    
    return {
        "code_snippet": code_snippet, 
        "language": language, 
        "label": label, 
        "timestamp": timestamp
    }

# Generate a dataset of code snippets with varying complexities
def generate_code_dataset(num_samples=1000):
    dataset = []
    
    complexities = ["beginner", "starter", "intermediate", "advanced", "pro", "master"]
    
    for _ in range(num_samples):
        complexity = random.choice(complexities)
        snippet = generate_code_snippet(complexity=complexity)
        dataset.append(snippet)
    
    return dataset

# Function to save dataset to CSV file
def save_to_csv(dataset, filename="code_dataset.csv"):
    keys = dataset[0].keys()
    with open(filename, mode="w", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(dataset)
    print(f"Dataset saved to {filename}")

# Function to save dataset to JSON file
def save_to_json(dataset, filename="code_dataset.json"):
    with open(filename, 'w') as file:
        json.dump(dataset, file, indent=4)
    print(f"Dataset saved to {filename}")

# Example: Generate a dataset of 20 code snippets with varying complexities
generated_data = generate_code_dataset(20)

# Save the dataset to CSV and JSON files
save_to_csv(generated_data, "code_dataset.csv")
save_to_json(generated_data, "code_dataset.json")
