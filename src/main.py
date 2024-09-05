import json
import pandas as pd
from training import Training

def load_config(config_path):
    """
    Loads the configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Loaded configuration dictionary.
    """
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def main():
    """
    Main function to set up and run the training process.
    """
    # Paths to configuration and data files
    config_path = 'config/config.json'
    data_path = 'data/data.csv'

    # Load model configuration
    model_config = load_config(config_path)

    # Initialize and run training
    training = Training(model_config, data_path)
    training.run_training()

if __name__ == "__main__":
    main()
