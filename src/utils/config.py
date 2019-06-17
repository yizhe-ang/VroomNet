import json


def process_config(json_file):
    """Load configuration as a python dictionary.
    
    Args:
        json_file (str): Path to the JSON file.
    
    Returns:
        dict:
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    
    return config_dict
