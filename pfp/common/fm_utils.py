import torch
import numpy as np
import torchvision.transforms as T
from pfp import DEVICE
import torch.nn as nn

def image_transform(size=224):
    return T.Compose([
            T.Resize((size, size)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

def fps_numpy(points: np.ndarray, n_samples: int):
    N, _ = points.shape
    sample_idx = np.zeros(n_samples, dtype=int)
    distances = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_samples):
        sample_idx[i] = farthest
        dist = np.sum((points - points[farthest]) ** 2, axis=-1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return sample_idx

def get_timesteps(schedule: str, k_steps: int, exp_scale: float = 1.0):
    t = torch.linspace(0, 1, k_steps + 1)[:-1]
    if schedule == "linear":
        dt = torch.ones(k_steps) / k_steps
    elif schedule == "cosine":
        dt = torch.cos(t * torch.pi) + 1
        dt /= torch.sum(dt)
    elif schedule == "exp":
        dt = torch.exp(-t * exp_scale)
        dt /= torch.sum(dt)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    t0 = torch.cat((torch.zeros(1), torch.cumsum(dt, dim=0)[:-1]))
    return t0, dt



def extract_dino_features_from_map(
    model: nn.Module, 
    images: torch.Tensor, 
    map_idx: torch.Tensor, 
    pixel_idx: torch.Tensor, 
    patch_size: int = 16
) -> torch.Tensor:
    """
    Extract DINOv2 patch features for selected pixels.

    Args:
        model: DINOv2 backbone
        images: (B, T, N, 3, H, W)
        map_idx: (K,) tensor of indices pointing to flattened images (B*T*N)
        pixel_idx: (K, 2) tensor of pixel coordinates (x, y) within the image
        patch_size: patch size used in DINOv2 (default 14)

    Returns:
        features: (K, C) tensor, DINOv2 features for each selected pixel
    """
    B, T, N, H, W, C_img = images.shape
    print("images shape: ", images.shape)
    K = map_idx.shape[2]
    images = images.reshape(-1, H, W, C_img)
    images = images.float()/255.0
    images = images.permute(0, 3, 1, 2).contiguous()
    transform = image_transform()
    images = transform(images)  # [B*T*N, 3, 224, 224]

    print("images shape: ", images.shape)
    # Flatten images along (B*T*N) dimension
    # images_flat = images_flat.to(DEVICE)
    with torch.no_grad():
        # Forward pass to get patch tokens
        tokens = model.forward_features(images)  # (B*T*N, HW, C)
    print("tokens shape: ", tokens.shape)
    Hp, Wp = H // patch_size, W // patch_size

    b_ids = torch.arange(B).view(B, 1, 1).expand(B, T, K).to(DEVICE) # (B,T,K)
    t_ids = torch.arange(T).view(1, T, 1).expand(B, T, K).to(DEVICE)  # (B,T,K)

    flat_img_idx = (b_ids * (T * N) + t_ids * N + map_idx).reshape(-1)  # (B*T*K)


    # Convert pixel coordinates to patch indices
    patch_y = pixel_idx[:, 0] // patch_size
    patch_x = pixel_idx[:, 1] // patch_size
    patch_flat_idx = patch_y * Wp + patch_x  # (K,)

    # Gather features
    features = tokens[flat_img_idx, patch_flat_idx]  # (K, C)
    C = features.shape[1]
    features = features.view(B, T, K, C)

    return features


