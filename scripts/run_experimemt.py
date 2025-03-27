import argparse
import yaml
import os

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_experiment(config):
    """Runs a training or evaluation experiment based on the config."""
    mode = config.get("mode", "train")  # Default mode is 'train'

    if mode == "train":
        os.system(f"python src/quantum_ml/train.py --config {config_path}")
    elif mode == "eval":
        os.system(f"python src/quantum_ml/evaluate.py --config {config_path}")
    else:
        raise ValueError("Invalid mode. Use 'train' or 'eval'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quantum-Classical AI Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load configuration and run
    config_path = args.config
    config = load_config(config_path)
    run_experiment(config)

