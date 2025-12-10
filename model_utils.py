# model_utils.py
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np

def get_model_and_transform(device="cpu"):
    # Novo padrão do torchvision
    weights = models.ResNet50_Weights.IMAGENET1K_V2

    # Carrega o modelo pré-treinado
    model = models.resnet50(weights=weights)

    # Remove a última camada (fc)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    # AQUI está a correção:
    # Usa os transforms prontos do weights (já inclui mean/std)
    transform = weights.transforms()

    return model, transform


def image_to_embedding(image_path_or_pil, model, transform, device="cpu"):
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(x).cpu().numpy()

    emb = emb.reshape(-1)

    # normalização
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb.astype("float32")