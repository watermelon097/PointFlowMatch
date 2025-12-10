import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pfp.common.upsample_anything import UPA

class DINOExtractor:
    def __init__(self, model_path, model_name, device='cuda', patch_size=16):
        self.device = device
        self.patch_size = patch_size
        
        # 加载模型
        print(f"Loading DINO from {model_path}...")
        self.model = torch.hub.load("./", model_name, source='local', weights=model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # DINO 标准预处理 (ImageNet Mean/Std)
        self.transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_and_sample(
        self, 
        rgb_maps: torch.Tensor, 
        pixels: torch.Tensor, 
        cam_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        输入:
            rgb_maps: (N_cam, H, W, 3) - 原始图像 uint8 或 float [0,1]
            pixels:   (N_points, 2)    - 采样点的像素坐标 (v, u) / (row, col)
            cam_ids:  (N_points,)      - 采样点属于哪个相机
        输出:
            features: (N_points, Embed_Dim) - 每个点对应的 DINO 特征
        """
        N_cam, H, W, C = rgb_maps.shape
        
        # 1. 预处理图像 for DINO
        # DINO 需要 (B, C, H, W)，且需要 Normalize
        # 假设 rgb_maps 是 [0, 1] 的 float，如果是 uint8 需要先除 255
        if rgb_maps.dtype == torch.uint8:
            imgs = rgb_maps.float() / 255.0
        else:
            imgs = rgb_maps
            
        imgs = imgs.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        # 应用 Normalize
        imgs = self.transform(imgs)

        # 2. Batch 推理 (一次跑 5 张图)
        # forward_features 返回字典，根据具体模型版本 key 可能不同
        # 通常 'x_norm_patchtokens' 是 patch 特征, 'x_norm_clstoken' 是 CLS
        output = self.model.forward_features(imgs)
        patch_feats = output["x_norm_patchtokens"] # (N_cam, N_patches, Embed_Dim)
               # 3. 采样 (Point-to-Feature Alignment)
        # 我们有 pixels (u, v) 在原图 H, W 坐标系
        # 由于图像被 resize 了，需要将坐标按比例缩放
        u = pixels[:, 1] # col
        v = pixels[:, 0] # row
        
        # 归一化公式: 2 * (x / (W-1)) - 1
        # 注意我们要映射到 feature map 的空间，但实际上直接用原图尺寸归一化即可，
        # 因为 grid_sample 是相对坐标。
        norm_x = 2.0 * (u / (W - 1)) - 1.0
        norm_y = 2.0 * (v / (H - 1)) - 1.0
        grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0).unsqueeze(0) # (1, 1, N_points, 2)
        
        # 这里的难点是：不同的点属于不同的 N_cam。
        # grid_sample 标准用法是 (N, C, H, W) 配 (N, H_out, W_out, 2)。
        # 我们的点是混合的。
        h_feat = H // self.patch_size
        w_feat = W // self.patch_size

        feat_maps = patch_feats.transpose(1, 2).reshape(N_cam, -1, h_feat, w_feat)
        
        hr_features = UPA(imgs, feat_maps)
        # 这种 gather 操作在 PyTorch 中写法：
        # 先把 maps 展平或者利用 fancy indexing
        point_features = feature_maps[cam_ids, :, feat_y, feat_x] # (N_points, Dim)
        
        return point_features