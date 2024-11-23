import argparse
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on an image using a pretrained segmentation model.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--output_dir', type=str, default="prediction", help="Directory to save the output.")
    return parser.parse_args()

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' does not exist.")
    
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b7", encoder_weights="imagenet", in_channels=3, classes=3)
    checkpoint = torch.load(checkpoint_path)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint file '{checkpoint_path}' does not contain 'model_state_dict'.")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, color in color_dict.items():
        output[mask == k] = color
    return output

def main():
    args = parse_args()

    model = load_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    color_dict = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}

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

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "segmented_image_color.png")
    cv2.imwrite(output_path, mask_rgb_bgr)

    print(f"Segmented image saved at {output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")