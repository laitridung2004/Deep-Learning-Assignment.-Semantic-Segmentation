import argparse
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

parser = argparse.ArgumentParser(description="Run inference on an image using a pretrained segmentation model.")
parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
args = parser.parse_args()

model = smp.UnetPlusPlus(encoder_name="efficientnet-b7", encoder_weights="imagenet", in_channels=3, classes=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('model.pth', map_location=device)
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])  
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

val_transform = A.Compose([A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(),])

color_dict = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}

def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, color in color_dict.items():
        output[mask == k] = color
    return output

os.makedirs("result", exist_ok=True)

image_path = args.image_path
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image path '{image_path}' does not exist.")

ori_img = cv2.imread(image_path)
if ori_img is None:
    raise ValueError(f"Failed to read the image at '{image_path}'.")

ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
ori_h, ori_w = ori_img.shape[:2]

img = cv2.resize(ori_img, (256, 256))
transformed = val_transform(image=img)
input_img = transformed["image"].unsqueeze(0).to(device)

with torch.no_grad():
    output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)

mask = cv2.resize(output_mask, (ori_w, ori_h))
mask = np.argmax(mask, axis=2)

mask_rgb = mask_to_rgb(mask, color_dict)
mask_rgb_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
output_path = os.path.join("result", "segmented_image_color.png")
cv2.imwrite(output_path, mask_rgb_bgr)

print(f"Segmented image saved at {output_path}")
