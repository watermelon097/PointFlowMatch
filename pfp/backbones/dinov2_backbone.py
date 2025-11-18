import torch
import torch.nn.functional as F
import sys
sys.path.append("/home/jun.wang/projects/PointFlowMatch/dinov2/")
import os
from dinov2.models.vision_transformer import vit_small




class DINOv2Backbone(torch.nn.Module):
    def __init__(self, model_name="dinov2_vits14", pretrained=True, ckpt_path=None):
        super().__init__()
        # 加载 DINOv2 from local package

        self.backbone = vit_small(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            block_chunks=0,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # new_h, new_w = model.patch_embed.grid_size  # or calculated from your input
        # old_h, old_w = 37, 37  # from the checkpoint, e.g., sqrt(1370-1) = 37
        # state_dict["pos_embed"] = self.resize_pos_embed(state_dict["pos_embed"], (new_h, new_w), (old_h, old_w), "bicubic")
        
        self.backbone.load_state_dict(state_dict, strict=True)
        
        
    def forward(self, x, resize_to=None):
        """
        x: [B, 3, H, W]
        resize_to: (H, W) - optional, interpolate to match point flow input resolution
        """
        print("input shape: ", x.shape)
        tokens = self.backbone(x)  # [B, N_patches, D]
        print(tokens.shape)
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)  # 假设 patch 正方形
        feat_map = tokens.transpose(1, 2).reshape(B, D, H, W)
        
        if resize_to:
            feat_map = F.interpolate(feat_map, size=resize_to, mode='bilinear', align_corners=False)
        return feat_map

    def forward_features(self, x):
        return self.backbone.forward_features(x)
