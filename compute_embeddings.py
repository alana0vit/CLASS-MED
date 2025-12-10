# compute_embeddings.py
from model_utils import get_model_and_transform, image_to_embedding
from db import load_all_embeddings
import numpy as np
# left as utility if needed