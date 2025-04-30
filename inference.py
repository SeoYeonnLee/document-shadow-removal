import os
import argparse
import time
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
import math
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.Shadow_matte import ShadowMattePredictor
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_image_overlap(img, crop_size, overlap_size):
    """Split large image into overlapping patches"""
    B, C, H, W = img.shape
    stride = crop_size - overlap_size
    
    y_starts = list(range(0, H - crop_size + 1, stride))
    if y_starts and y_starts[-1] != H - crop_size:
        y_starts.append(H - crop_size)
    
    x_starts = list(range(0, W - crop_size + 1, stride))
    if x_starts and x_starts[-1] != W - crop_size:
        x_starts.append(W - crop_size)

    patches = []
    positions = []
    for y in y_starts:
        for x in x_starts:
            patch = img[:, :, y:y+crop_size, x:x+crop_size]
            patches.append(patch)
            positions.append((y, x, crop_size, crop_size))
    
    return patches, positions

def create_gaussian_mask(patch_size, overlap):
    """Create gaussian blending mask for overlapping patches"""
    h, w = patch_size
    weight_y = torch.ones(h, dtype=torch.float32)
    sigma = overlap / 2.0 if overlap > 0 else 1.0

    for i in range(h):
        if i < overlap:
            weight_y[i] = math.exp(-0.5 * ((overlap - i)/sigma)**2)
        elif i > h - overlap - 1:
            weight_y[i] = math.exp(-0.5 * ((i - (h - overlap - 1))/sigma)**2)

    weight_x = torch.ones(w, dtype=torch.float32)
    for j in range(w):
        if j < overlap:
            weight_x[j] = math.exp(-0.5 * ((overlap - j)/sigma)**2)
        elif j > w - overlap - 1:
            weight_x[j] = math.exp(-0.5 * ((j - (w - overlap - 1))/sigma)**2)

    mask = torch.ger(weight_y, weight_x)
    return mask.unsqueeze(0).unsqueeze(0)

def merge_image_overlap(patches, positions, resolution, overlap_size, blend_mode='gaussian'):
    """Merge overlapping patches back into a single image"""
    B, C, H, W = resolution
    device = patches[0].device

    merged = torch.zeros((B, C, H, W), device=device)
    weight_sum = torch.zeros((B, 1, H, W), device=device)

    for patch, pos in zip(patches, positions):
        y, x, ph, pw = pos

        if blend_mode == 'gaussian' and overlap_size > 0:
            mask = create_gaussian_mask((ph, pw), overlap_size).to(device)
        else:
            mask = torch.ones((1, 1, ph, pw), device=device)

        merged[:, :, y:y+ph, x:x+pw] += patch * mask
        weight_sum[:, :, y:y+ph, x:x+pw] += mask

    merged = merged / (weight_sum + 1e-8)
    return merged

class InferenceDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.image_paths = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path)) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
            
        return image, img_name, image.shape

def save_image(tensor, filename, output_dir):
    tensor = tensor.cpu().detach()
    
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor[0]
        
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor.permute(1, 2, 0).numpy()

    img = Image.fromarray((tensor * 255).astype(np.uint8))

    img.save(os.path.join(output_dir, filename))

def parse_args():
    parser = argparse.ArgumentParser(description="Shadow Removal Inference with ViT + Shadow Matte")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained ViT model")
    parser.add_argument('--shadow_matte_path', type=str, default="./checkpoints/UNet/image_512/epoch180.pth", help="Path to the shadow matte model")
    parser.add_argument('--input_dir', type=str, default='../data/Kligler/test/input', help="Directory containing input images")
    parser.add_argument('--output_dir', type=str, default='./results', help="Directory to save output images")
    parser.add_argument('--patch_size', type=int, default=256, help="Size of image patches for processing")
    parser.add_argument('--overlap', type=int, default=32, help="Overlap size between patches")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for patch processing")
    parser.add_argument('--save_matte', action='store_true', help="Whether to save shadow matte predictions")
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_matte:
        os.makedirs(os.path.join(args.output_dir, 'mattes'), exist_ok=True)

    matte_predictor = ShadowMattePredictor(
        model_path=args.shadow_matte_path,
        img_size=args.patch_size,
        device=device
    )

    model = MaskedAutoencoderViT(
        patch_size=8, 
        in_chans=4, 
        out_chans=3, 
        embed_dim=256, 
        depth=6, 
        num_heads=8,
        decoder_embed_dim=256, 
        decoder_depth=6, 
        decoder_num_heads=8,
        mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    model.to(device)

    print(f"Loading model weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    dataset = InferenceDataset(args.input_dir)
    
    print(f"Processing {len(dataset)} images...")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Processing images"):
            input_img, filename, _ = dataset[i]
            
            input_img = input_img.unsqueeze(0).to(device)

            patches, positions = split_image_overlap(
                input_img, 
                crop_size=args.patch_size, 
                overlap_size=args.overlap
            )

            all_outputs = []
            all_mattes = []
            
            for j in range(0, len(patches), args.batch_size):
                batch_patches = torch.cat(patches[j:j+args.batch_size], dim=0)
                
                shadow_mattes = matte_predictor.predict(batch_patches, is_tensor=True)
                shadow_mattes = shadow_mattes.to(device)

                combined_input = torch.cat([batch_patches, shadow_mattes], dim=1)

                batch_outputs = model(combined_input)
                
                for k in range(batch_outputs.shape[0]):
                    all_outputs.append(batch_outputs[k:k+1])
                    all_mattes.append(shadow_mattes[k:k+1])
            
            output_img = merge_image_overlap(
                all_outputs, 
                positions, 
                resolution=input_img.shape, 
                overlap_size=args.overlap, 
                blend_mode='gaussian'
            )

            save_image(output_img, filename, args.output_dir)

            if args.save_matte:
                matte_img = merge_image_overlap(
                    all_mattes, 
                    positions, 
                    resolution=(1, 1, input_img.shape[2], input_img.shape[3]), 
                    overlap_size=args.overlap, 
                    blend_mode='gaussian'
                )
                matte_filename = f"matte_{filename}"
                save_image(matte_img, matte_filename, os.path.join(args.output_dir, 'mattes'))

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Inference completed in {elapsed_time:.2f} seconds")