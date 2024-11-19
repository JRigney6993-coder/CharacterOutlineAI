import json

def load_config(config_path="config.json"):
    """Load configuration from the JSON file."""
    with open(config_path, "r") as file:
        return json.load(file)
