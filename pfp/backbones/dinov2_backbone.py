import torch
import torch.nn.functional as F

class DINOv2Backbone(torch.nn.Module):
    def __init__(self, model_name="dinov2_vitl14", pretrained=True, ckpt_path=None):
        super().__init__()
        # 加载 DINOv2
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=pretrained
        )
        
        # 如果提供 checkpoint，覆盖权重
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = {k: v for k, v in ckpt.items() if k in self.backbone.state_dict()}
            self.backbone.load_state_dict(state_dict)
        
    def forward(self, x, resize_to=None):
        """
        x: [B, 3, H, W]
        resize_to: (H, W) - optional, interpolate to match point flow input resolution
        """
        tokens = self.backbone(x)  # [B, N_patches, D]
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)  # 假设 patch 正方形
        feat_map = tokens.transpose(1, 2).reshape(B, D, H, W)
        
        if resize_to:
            feat_map = F.interpolate(feat_map, size=resize_to, mode='bilinear', align_corners=False)
        return feat_map
