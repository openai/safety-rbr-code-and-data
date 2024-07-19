import matplotlib.pyplot as plt
import numpy as np
import json
import yaml

#Util for reading in Jsonl files
def read_jsonl(file_path):
    """
    Reads a JSONL (JSON Lines) file and returns a list of JSON-serializable objects.

    Args:
        Path to the JSONL file.
    
    Returns:
        List of JSON-serializable objects.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    
    return data

#Util for turning conversations to strings for hashing
def to_string(conversation):
  convo_parts = [f"|{message['role'].upper()}|: {message['content']}" for message in conversation]
  convo_str = "\n".join(convo_parts)
  return convo_str

#Util for reading in YAML files
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)
    return content

#Some Utils for plotting/getting nicer label names for plots
def get_colors(num_samples = 3, end_point = 1.0):
  sample_points = np.linspace(0.1, end_point, num_samples)
  return plt.cm.viridis(sample_points)

colors = get_colors(num_samples = 5)
colors2 = get_colors(num_samples = 3, end_point=0.9)
color_mapping = {
    'Comply': {
        'fully complies': colors[0],
        'partially complies': colors[1],
        'random non-refusal': colors[2],
        'imperfect refusal': colors[3],
        'perfect refusal': colors[4],
    },
    'Hard Refuse': {
        'perfect hard refusal': colors2[0],
        'bad hard refusal': colors2[1],
        'disallowed content': colors2[2]
    },
    'Safe Refuse': {
        'perfect safe completion': colors2[0],
        'bad safe completion': colors2[1],
        'disallowed content': colors2[2],
    },
}

labels_map = {
    'fully complies': "Complies (Ideal)",
    'partially complies': "Partially Complies",
    'random non-refusal': "Random Non-Refusal",
    'imperfect refusal': "Bad Refusal",
    'perfect refusal': "Perfect Refusal",
    'perfect hard refusal': "Perfect Refusal (Ideal)",
    'bad hard refusal': "Bad Refusal",
    'disallowed content': "Disallowed",
    'perfect safe completion': "Perfect Safe Refusal (Ideal)",
    'bad safe completion': "Bad Safe Refusal"
}