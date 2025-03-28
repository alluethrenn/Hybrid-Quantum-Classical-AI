import os
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.datasets import load_iris, load_digits
from datasets import load_dataset
import joblib  # To save models/datasets as Pickle files


# Ensure save directories exist
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Function to save scikit-learn datasets as CSV
def save_sklearn_datasets():
    iris = load_iris()
    digits = load_digits()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df.to_csv('./data/iris.csv', index=False)

    digits_df = pd.DataFrame(data=digits.data, columns=digits.feature_names)
    digits_df['target'] = digits.target
    digits_df.to_csv('./data/digits.csv', index=False)

    print("Scikit-learn datasets saved as CSV!")


# Function to save torchvision datasets as Images or CSV
def save_torchvision_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Save CIFAR-10 images as CSV (flattened)
    cifar10_data = [np.array(image).flatten() for image, _ in cifar10]
    cifar10_labels = [label for _, label in cifar10]
    cifar10_df = pd.DataFrame(cifar10_data)
    cifar10_df['label'] = cifar10_labels
    cifar10_df.to_csv('./data/cifar10.csv', index=False)

    # Save MNIST images as CSV (flattened)
    mnist_data = [np.array(image).flatten() for image, _ in mnist]
    mnist_labels = [label for _, label in mnist]
    mnist_df = pd.DataFrame(mnist_data)
    mnist_df['label'] = mnist_labels
    mnist_df.to_csv('./data/mnist.csv', index=False)

    print("Torchvision datasets saved as CSV!")


# Function to save tensorflow-datasets as CSV
def save_tf_datasets():
    mnist = tfds.load('mnist', split='train')
    cifar10 = tfds.load('cifar10', split='train')

    # Convert datasets to pandas DataFrames and save as CSV
    mnist_df = pd.DataFrame([example['image'].numpy().flatten() for example in mnist])
    mnist_df['label'] = [example['label'].numpy() for example in mnist]
    mnist_df.to_csv('./data/mnist_tf.csv', index=False)

    cifar10_df = pd.DataFrame([example['image'].numpy().flatten() for example in cifar10])
    cifar10_df['label'] = [example['label'].numpy() for example in cifar10]
    cifar10_df.to_csv('./data/cifar10_tf.csv', index=False)

    print("Tensorflow datasets saved as CSV!")


# Function to save Hugging Face datasets as CSV
def save_huggingface_datasets():
    imdb = load_dataset('imdb', split='train')
    cifar10 = load_dataset('cifar10', split='train')

    # Save IMDB dataset as CSV (just the text and labels for simplicity)
    imdb_df = pd.DataFrame({
        'text': [example['text'] for example in imdb],
        'label': [example['label'] for example in imdb]
    })
    imdb_df.to_csv('./data/imdb.csv', index=False)

    # Save CIFAR-10 dataset as CSV (flattened)
    cifar10_data = [np.array(example['img']).flatten() for example in cifar10]
    cifar10_labels = [example['label'] for example in cifar10]
    cifar10_df = pd.DataFrame(cifar10_data)
    cifar10_df['label'] = cifar10_labels
    cifar10_df.to_csv('./data/cifar10_hf.csv', index=False)

    print("Hugging Face datasets saved as CSV!")


# Function to download and save Kaggle dataset
def download_and_save_kaggle_dataset():
    kaggle.api.dataset_download_files('zillow/zecon', path='./data', unzip=True)
    print("Kaggle dataset downloaded and saved!")


# Main function to download and save datasets
def download_and_save_all_datasets():
    ensure_directory('./data')  # Ensure data directory exists
    
    save_sklearn_datasets()
    save_torchvision_datasets()
    save_tf_datasets()
    save_huggingface_datasets()
    download_and_save_kaggle_dataset()


# Run the main function to download and save all datasets
download_and_save_all_datasets()
