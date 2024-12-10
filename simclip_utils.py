import numpy as np
import yaml

# --- Utility Functions ---
def is_photo(file_path):
    """Check if the file is a photo (JPG or PNG)."""
    return file_path.lower().endswith(('.jpg', '.png'))


def make_mosaic(images):
    """Create a 2x2 mosaic from 4 images."""
    row1 = np.hstack([images[0], images[1]])
    row2 = np.hstack([images[2], images[3]])
    return np.vstack([row1, row2])


# --- Load Configuration ---
def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config