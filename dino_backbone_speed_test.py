import torch
from pfp.backbones.dinov2_backbone import DINOv2Backbone
import torchvision.transforms as T
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F


img = Image.open("vis/rgb_obs_10_0.png")

show_img = False
image_transform = T.Compose([
    T.Resize((224, 224)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),


])

img_tensor = image_transform(img)
# Display original image and transformed image tensor (denormalize for visualization)
img_np = img_tensor.permute(1, 2, 0).numpy()
img_np = (img_np * 0.5 + 0.5).clip(0, 1)  # Denormalize

if show_img: # Display both images side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    axes[1].imshow(img_np)
    axes[1].axis('off')
    axes[1].set_title('Transformed Image Tensor')
    plt.tight_layout()
    plt.show()


# 加载 ViT-L/14 模型
model = DINOv2Backbone(
    ckpt_path="./ckpt/dinov2_vits14_pretrain.pth"
)

model.eval()
model.cuda()



with torch.no_grad():
    feats = model.backbone(img_tensor.unsqueeze(-1).cuda())
    patch_tokens = model.backbone.forward_features(img_tensor.unsqueeze(0).cuda())
    patch_tokens = patch_tokens["x_norm_patchtokens"].cpu()

    print(feats.shape)
    patch_feats = feats[:,:,:]
    h = w = int(patch_feats.shape[1] ** 0.5)
    feat_map = patch_feats[0].reshape(h,w,-1).permute(2,0,1)

    print("feat_map shape: ", feat_map.shape)

heat = patch_tokens.norm(dim=-1)
heatmap  = heat.reshape(1,16,16)

heatmap = F.interpolate(heatmap.unsqueeze(0), size=(img.height, img.width), mode='bilinear')[0,0]

# plot heatmap
plt.imshow(heatmap.cpu().numpy())
plt.show()

# heatmap = feat_map.norm(dim=0)
# heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
# heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(img.height, img.width), mode='bilinear')[0,0]
# print("heatmap shape: ", heatmap.shape)

# img_cv = np.array(img)
# heatmap_np = (heatmap.cpu().numpy()*255.0).astype(np.uint8)
# heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
# overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Heatmap")
# plt.imshow(heatmap_color[...,::-1])
# plt.subplot(1, 2, 2)
# plt.title("Overlay")
# plt.imshow(overlay[...,::-1])
# plt.show()
