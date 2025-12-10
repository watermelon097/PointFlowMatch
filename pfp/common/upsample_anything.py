import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

# ---------------------------------------------------------
# 1. 修改后的 Gather 函数，支持 Batch 索引
# ---------------------------------------------------------
def gather_lr_batch(map_lr: torch.Tensor, Ui: torch.Tensor, Vi: torch.Tensor):
    """
    Args:
        map_lr: [B, C, Hl, Wl]
        Ui, Vi: [Bn, Hh, Wh] - 邻域偏移索引
    Returns:
        [B, C, Bn, Hh, Wh] - Gather 后的值
    """
    # PyTorch 的 Advanced Indexing 允许我们直接使用 Ui, Vi 进行索引
    # map_lr[..., Ui, Vi] 会自动广播 Batch 和 Channel 维度
    # 结果形状: [B, C, Bn, Hh, Wh]
    return map_lr[..., Ui, Vi]

@torch.no_grad()
def _build_offsets(R_max: int, device: torch.device):
    """返回方形半径 R_max 内的邻居偏移量。"""
    offs = torch.arange(-R_max, R_max + 1, device=device)
    dY, dX = torch.meshgrid(offs, offs, indexing='ij')
    return dY.reshape(-1), dX.reshape(-1)

def _tanh_bound_pi(raw: torch.Tensor):
    return math.pi * torch.tanh(raw)

# ---------------------------------------------------------
# 2. 修改后的核心 JBU 算法，支持 Batch
# ---------------------------------------------------------
def gs_jbu_aniso_noparent(
    feat_lr: torch.Tensor,     # [B,C,Hl,Wl]
    guide_hr: torch.Tensor,    # [B,3,Hh,Wh]
    scale: int,
    sigma_x_map: torch.Tensor, # [B,1,Hl,Wl]
    sigma_y_map: torch.Tensor, # [B,1,Hl,Wl]
    theta_map: torch.Tensor,   # [B,1,Hl,Wl]
    sigma_r_map: torch.Tensor, # [B,1,Hl,Wl]
    R_max: int = 4,
    alpha_dyn: float = 2.0,
    C_chunk: int = 512,
    Nn_chunk: int = 81,
    center_mode: str = "nearest",
    use_autocast: bool = True,
):
    # 获取 Batch Size (B)
    B, C, Hl, Wl = feat_lr.shape
    _, _, Hh, Wh = guide_hr.shape

    dev = feat_lr.device
    dtype_feat = feat_lr.dtype
    dtype_acc = torch.float32

    # 坐标网格生成 (与 Batch 无关)
    y = torch.arange(Hh, device=dev, dtype=torch.float32)
    x = torch.arange(Wh, device=dev, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    u = (Y + 0.5) / scale - 0.5
    v = (X + 0.5) / scale - 0.5

    if center_mode == "nearest":
        uc = torch.round(u).clamp(0, Hl - 1).to(torch.long)
        vc = torch.round(v).clamp(0, Wl - 1).to(torch.long)
    elif center_mode == "floor":
        uc = torch.floor(u).clamp(0, Hl - 1).to(torch.long)
        vc = torch.floor(v).clamp(0, Wl - 1).to(torch.long)
    else:
        raise ValueError("center_mode must be 'nearest' or 'floor'")

    # 动态半径计算 (支持 Batch)
    sigma_eff = torch.maximum(sigma_x_map, sigma_y_map)  # [B,1,Hl,Wl]
    sigma_eff_hr = F.interpolate(sigma_eff, (Hh, Wh), mode='bilinear', align_corners=False)
    # 取 Batch 中最大的半径图以构建统一掩码 (或者可以在循环内分别处理，这里为了性能取 max)
    # Clamp R_map to handle constraints
    R_map = torch.ceil(alpha_dyn * sigma_eff_hr).clamp_(min=1, max=R_max).to(torch.int64) # [B,1,Hh,Wh]
    
    # 这里的 mask 逻辑稍微复杂，为了简化并行计算，我们可以取 Batch 中的最大半径作为循环依据，
    # 或者直接使用 R_max。在内部 mask 判定时，R_map 会广播。
    
    dY_all, dX_all = _build_offsets(R_max, dev)
    K = dY_all.numel()

    # 初始化累加器: 增加 Batch 维度 B
    num_s = torch.zeros(B, C, Hh, Wh, device=dev, dtype=dtype_acc)
    den_s = torch.zeros(B, 1, Hh, Wh, device=dev, dtype=dtype_acc)
    # m 用于 LogSumExp trick 的数值稳定性
    m     = torch.full((B, 1, Hh, Wh), float("-inf"), device=dev, dtype=dtype_acc)

    guide32 = guide_hr.to(torch.float32, copy=False)
    
    # 将参数转为 float32
    sx_map32 = sigma_x_map.to(torch.float32, copy=False)
    sy_map32 = sigma_y_map.to(torch.float32, copy=False)
    th_map32 = theta_map.to(torch.float32, copy=False)
    sr_map32 = sigma_r_map.to(torch.float32, copy=False)

    # 对 Guide Image 进行下采样，用于计算 spatial weight 中的 intensity term
    guide_lr = F.interpolate(guide32, size=(Hl, Wl), mode='bilinear', align_corners=False)  # [B,3,Hl,Wl]

    autocast_ctx = torch.cuda.amp.autocast(enabled=use_autocast, dtype=torch.float16)

    with autocast_ctx:
        for n0 in range(0, K, Nn_chunk):
            n1 = min(n0 + Nn_chunk, K)
            dY = dY_all[n0:n1].view(-1, 1, 1)  # [Bn,1,1]
            dX = dX_all[n0:n1].view(-1, 1, 1)
            Bn = dY.shape[0]

            # Ui, Vi: [Bn, Hh, Wh] (Batch共享相同的空间偏移)
            Ui = torch.clamp(uc.unsqueeze(0) + dY, 0, Hl - 1)
            Vi = torch.clamp(vc.unsqueeze(0) + dX, 0, Wl - 1)

            rad2 = (dY ** 2 + dX ** 2)
            # R_map 是 [B,1,Hh,Wh]，需要广播去比较 [Bn,1,1]
            # 结果 mask: [B, Bn, Hh, Wh]
            mask = (rad2.unsqueeze(0) <= (R_map ** 2)) # [B, Bn, Hh, Wh]

            cy = (Ui.to(torch.float32) + 0.5) * scale - 0.5
            cx = (Vi.to(torch.float32) + 0.5) * scale - 0.5
            dx = X.unsqueeze(0) - cx # [Bn, Hh, Wh]
            dy = Y.unsqueeze(0) - cy

            # 使用 batch gather
            # 结果形状: [B, 1, Bn, Hh, Wh] -> squeeze(1) -> [B, Bn, Hh, Wh]
            sx = gather_lr_batch(sx_map32, Ui, Vi).squeeze(1).clamp_min(1e-6)
            sy = gather_lr_batch(sy_map32, Ui, Vi).squeeze(1).clamp_min(1e-6)
            th = gather_lr_batch(th_map32, Ui, Vi).squeeze(1)
            sr = gather_lr_batch(sr_map32, Ui, Vi).squeeze(1).clamp_min(1e-6)

            cos_t, sin_t = torch.cos(th), torch.sin(th)
            # dx, dy 是 [Bn, Hh, Wh]，广播到 [B, Bn, Hh, Wh]
            x_p = dx.unsqueeze(0) * cos_t + dy.unsqueeze(0) * sin_t
            y_p = -dx.unsqueeze(0) * sin_t + dy.unsqueeze(0) * cos_t
            
            # 空间权重
            log_ws = -(x_p ** 2) / (2 * sx ** 2 + 1e-8) - (y_p ** 2) / (2 * sy ** 2 + 1e-8)

            # 颜色引导权重
            # Guide Channels: [B, 3, Hl, Wl]
            # Gathered: [B, 1, Bn, Hh, Wh] for each channel
            g0 = gather_lr_batch(guide_lr[:, 0:1, ...], Ui, Vi).squeeze(1)
            g1 = gather_lr_batch(guide_lr[:, 1:2, ...], Ui, Vi).squeeze(1)
            g2 = gather_lr_batch(guide_lr[:, 2:3, ...], Ui, Vi).squeeze(1)
            
            # Guide HR: [B, 3, Hh, Wh] -> unsqueeze(1) -> [B, 3, 1, Hh, Wh] 用于广播 Bn
            diff2 = (guide32[:, 0:1, ...].unsqueeze(2) - g0.unsqueeze(1)) ** 2 + \
                    (guide32[:, 1:2, ...].unsqueeze(2) - g1.unsqueeze(1)) ** 2 + \
                    (guide32[:, 2:3, ...].unsqueeze(2) - g2.unsqueeze(1)) ** 2
            diff2 = diff2.squeeze(1) # [B, Bn, Hh, Wh]

            log_wr = -diff2 / (2.0 * sr * sr + 1e-8)

            log_w = log_ws + log_wr
            # 应用 Mask
            log_w = torch.where(mask, log_w, torch.full_like(log_w, float("-inf")))

            # LogSumExp 稳定性处理
            # max over Bn dimension (dim=1)
            m_chunk = torch.max(log_w, dim=1).values.unsqueeze(1) # [B, 1, Hh, Wh]
            
            valid = torch.isfinite(m_chunk)
            # 如果整个 Batch 都没有有效值，跳过 (通常不会发生，除非mask全False)
            if not valid.any():
                continue

            m_new = m.clone()
            # 更新最大值
            m_new[valid] = torch.maximum(m[valid], m_chunk[valid])

            delta = (m - m_new).clamp_max(0)
            scale_old = torch.exp(delta) # [B, 1, Hh, Wh]
            
            den_s.mul_(scale_old)
            num_s.mul_(scale_old) # scale_old 广播到 C

            log_w_shift = log_w - m_new # 广播减法
            # 处理无效值防止 NaN
            # log_w_shift 可能会有 -inf - (-inf) = NaN，所以需要mask
            # 但这里我们主要关心 valid 的部分。
            # 简单起见，重新 mask
            log_w_shift = torch.where(mask, log_w_shift, torch.tensor(float("-inf"), device=dev))
            
            s = torch.exp(log_w_shift)  # [B, Bn, Hh, Wh]
            # 此时 s 已经包含了无效邻居为 0 的处理

            den_s.add_(s.sum(dim=1, keepdim=True)) # sum over Bn

            # 特征聚合
            for c0 in range(0, C, C_chunk):
                c1 = min(c0 + C_chunk, C)
                # feat_lr: [B, C, Hl, Wl]
                # Gather feature chunk: [B, Cc, Bn, Hh, Wh]
                # 使用我们新的 gather 函数
                feat_sel = gather_lr_batch(feat_lr[:, c0:c1, ...], Ui, Vi)
                
                # Weighted Sum: sum over Bn (dim=2)
                # s: [B, Bn, Hh, Wh] -> unsqueeze(1) -> [B, 1, Bn, Hh, Wh]
                weighted_feat = (feat_sel * s.unsqueeze(1)).sum(dim=2)
                
                num_s[:, c0:c1].add_(weighted_feat)

            m = m_new

    out_raw = (num_s / den_s.clamp_min(1e-8)).to(dtype_feat)
    fallback = F.interpolate(feat_lr, size=(Hh, Wh), mode='bilinear', align_corners=False)
    tiny = (den_s < 1e-6) # [B, 1, Hh, Wh]
    out = torch.where(tiny, fallback, out_raw)
    return out


# ---------------------------------------------------------
# 3. 修改后的模型类，初始化时接受 batch_size
# ---------------------------------------------------------
class LearnablePixelwiseAnisoJBU_NoParent(nn.Module):

    def __init__(
        self,
        batch_size: int,   # 新增参数
        Hl: int,
        Wl: int,
        scale: int = 16,
        init_sigma: float = 16.0,
        init_sigma_r: float = 0.12,
        R_max: int = 8,
        alpha_dyn: float = 2.0,
        center_mode: str = "nearest",
        eval_C_chunk: int = 128,   
        eval_Nn_chunk: int = 49,   
        use_autocast: bool = True,
    ):
        super().__init__()
        self.scale = int(scale)
        self.R_max = int(R_max)
        self.alpha_dyn = float(alpha_dyn)
        self.center_mode = center_mode
        self.eval_C_chunk = int(eval_C_chunk)
        self.eval_Nn_chunk = int(eval_Nn_chunk)
        self.use_autocast = bool(use_autocast)

        # 参数形状修改为 [B, 1, Hl, Wl]
        self.sx_raw = nn.Parameter(torch.full((batch_size, 1, Hl, Wl), float(np.log(init_sigma)),   dtype=torch.float32))
        self.sy_raw = nn.Parameter(torch.full((batch_size, 1, Hl, Wl), float(np.log(init_sigma)),   dtype=torch.float32))
        self.th_raw = nn.Parameter(torch.zeros( (batch_size, 1, Hl, Wl),                             dtype=torch.float32))
        self.sr_raw = nn.Parameter(torch.full((batch_size, 1, Hl, Wl), float(np.log(init_sigma_r)), dtype=torch.float32))

    def forward(self, feat_lr: torch.Tensor, guide_hr: torch.Tensor):
        """
        feat_lr:  [B, C, Hl, Wl]
        guide_hr: [B, 3, Hh, Wh]
        """
        # 参数已经是 Batched 的了
        sigma_x = torch.exp(self.sx_raw)                 
        sigma_y = torch.exp(self.sy_raw)                 
        theta   = _tanh_bound_pi(self.th_raw)            
        sigma_r = torch.exp(self.sr_raw)                 

        C = int(feat_lr.shape[1])
        K = int((2 * self.R_max + 1) * (2 * self.R_max + 1))
        if self.training:
            C_chunk = C
            Nn_chunk = K
        else:
            C_chunk = min(self.eval_C_chunk, C)
            Nn_chunk = min(self.eval_Nn_chunk, K)

        return gs_jbu_aniso_noparent(
            feat_lr, guide_hr, self.scale,
            sigma_x, sigma_y, theta, sigma_r,
            R_max=self.R_max, alpha_dyn=self.alpha_dyn,
            C_chunk=C_chunk, Nn_chunk=Nn_chunk,
            center_mode=self.center_mode,
            use_autocast=self.use_autocast,
        )

# ---------------------------------------------------------
# 4. 修改后的 UPA 入口函数，处理 Batch 输入
# ---------------------------------------------------------
def UPA(HR_img_batch, lr_modality_batch):
    """
    Args:
        HR_img_batch: [B, 3, H, W]  (Float Tensor, normalized 0-1)
        lr_modality_batch: [B, C, Hl, Wl] (Feature map)
    """
    USE_AMP = True
    AMP_DTYPE = torch.float16
    
    # 确保输入在 GPU
    hr = HR_img_batch.cuda()
    lr_modaliry = lr_modality_batch.cuda()
    
    B, _, H, W = hr.shape
    _, _, Hl, Wl = lr_modaliry.shape
    scale = int(H/Hl)
    
    # 对 HR 进行下采样作为初始输入 (Loss 计算需要)
    # 注意：这里我们计算的是 HR 的 LR 版本，用于监督信号的一部分或者仅用于逻辑对齐
    # 但在原始代码中，input 'lr' to model is downsampled HR. 
    # The 'lr_modaliry' is only used in inference. 
    # 等等，原始代码逻辑是：Train mapping from LR-RGB to HR-RGB, then Apply to LR-Feat.
    
    # 训练阶段输入: Downsampled HR RGB
    lr_rgb = F.interpolate(hr, scale_factor=1/scale, mode="bicubic", align_corners=False)

    # 初始化模型，传入 Batch Size
    model = LearnablePixelwiseAnisoJBU_NoParent(batch_size=B, Hl=Hl, Wl=Wl, scale=scale).cuda()

    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    max_steps = 100 # 原始是5100，这里为了演示速度设为100，实际使用请改回 500-1000 左右
    
    # Gamma calculation requires non-zero division if max_steps is small
    gamma = (1e-9 / 1e-1) ** (1.0 / max_steps)
    scheduler = LambdaLR(opt, lr_lambda=lambda step: gamma ** step)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    for step in range(max_steps + 1):
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            # Forward: 输入 LR RGB, Guide HR RGB -> 预测 HR RGB
            pred = model(lr_rgb, hr)  # [B, 3, H, W]
            loss = F.l1_loss(pred, hr) # L1 Loss 支持 Batch 自动平均

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        scheduler.step()
        
        # 原始代码有 early break at 50? 
        if step == 50: 
            break

    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
        # Inference: 输入 LR Feature, Guide HR RGB -> 预测 HR Feature
        hr_feat = model(lr_modaliry, hr)
        
    return hr_feat