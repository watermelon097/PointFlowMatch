import torch
import torch.nn.functional as F
import torchvision.transforms as T

class DINOExtractor:
    def __init__(
        self, 
        model_path: str = './ckpt/dinov3_vits16.pth',
        model_name: str = 'dinov3_vits16',
        device: str = 'cuda',
        patch_size: int = 16
    ):
        self.device = device
        self.patch_size = patch_size
        
        # 加载模型
        print(f"Loading DINO from {model_path}...")
        self.model = torch.hub.load("../dinov3", model_name, source='local', weights=model_path)
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

        imgs = torch.from_numpy(rgb_maps).to(self.device, dtype=torch.float32) / 255.0
        imgs = imgs.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        # 应用 Normalize
        imgs = self.transform(imgs)

        # 2. Batch 推理 (一次跑 5 张图)
        # forward_features 返回字典，根据具体模型版本 key 可能不同
        # 通常 'x_norm_patchtokens' 是 patch 特征, 'x_norm_clstoken' 是 CLS
        output = self.model.forward_features(imgs)
        patch_feats = output["x_norm_patchtokens"] # (N_cam, N_patches, Embed_Dim)
        # 3. 采样 (Point-to-Feature Alignment)
        # 计算 feature map 的尺寸
        h_feat = H // self.patch_size
        w_feat = W // self.patch_size
        
        # (N_cam, N_patches, Embed_Dim) -> (N_cam, Embed_Dim, h_feat, w_feat)
        embed_dim = patch_feats.shape[-1]
        feat_maps = patch_feats.reshape(N_cam, h_feat, w_feat, embed_dim).permute(0, 3, 1, 2)
        
        # 从 pixels 提取坐标
        # pixels: (N_points, 2) - (v, u) / (row, col)
        u = pixels[:, 1]  # col (x坐标)
        v = pixels[:, 0]  # row (y坐标)
        
        # 将像素坐标转换为 feature map 坐标
        # feature map 的每个位置对应 patch_size x patch_size 的像素区域
        feat_x = (u / self.patch_size).long().clamp(min=0, max=w_feat - 1)
        feat_y = (v / self.patch_size).long().clamp(min=0, max=h_feat - 1)
        
        # 使用 cam_ids 和计算出的坐标提取特征
        # feat_maps: (N_cam, Embed_Dim, h_feat, w_feat)
        # cam_ids: (N_points,)
        # feat_y, feat_x: (N_points,)
        # 使用 advanced indexing: feat_maps[cam_ids, :, feat_y, feat_x]
        point_features = feat_maps[cam_ids, :, feat_y, feat_x]  # (N_points, Embed_Dim)
        
        return point_features