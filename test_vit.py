import argparse
import logging
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.UTILS import compute_psnr, compute_rmse, compute_ssim
from datasets.datasets_pairs import my_dataset_eval
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.Shadow_matte import ShadowMattePredictor

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(description="Evaluate trained ViT model for shadow removal")
parser.add_argument('--model_path', type=str, required=True, help="Path to trained model weights")
parser.add_argument('--test_input_dir', type=str, default='../data/Kligler/test/input', help="Directory containing test input images")
parser.add_argument('--test_target_dir', type=str, default='../data/Kligler/test/target', help="Directory containing test target images")
parser.add_argument('--output_dir', type=str, default='./test_results', help="Directory to save output images")
parser.add_argument('--img_size', type=int, default=256, help="Image size for evaluation")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for evaluation")
parser.add_argument('--shadow_matte_path', type=str, default="../Pytorch-UNet/checkpoints/image_512/epoch180.pth", help="Path to shadow matte model weights")
parser.add_argument('--save_outputs', action='store_true', help="Save output images")
args = parser.parse_args()

if args.save_outputs:
    os.makedirs(args.output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')) if args.save_outputs else logging.NullHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def main():
    test_dataset = my_dataset_eval(
        root_in=args.test_input_dir,
        root_label=args.test_target_dir,
        fix_sample=10000 
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    logger.info(f"Test dataset size: {len(test_dataset)}")

    matte_predictor = ShadowMattePredictor(
        model_path=args.shadow_matte_path,
        img_size=args.img_size,
        device=device
    )
    logger.info(f"Shadow matte predictor loaded from {args.shadow_matte_path}")

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
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    logger.info(f"Model loaded from {args.model_path}")

    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    processing_times = []

    with torch.no_grad():
        for i, (input_img, target_img, img_name) in enumerate(test_loader):
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            start_time = time.time()
            shadow_matte = matte_predictor.predict(input_img, is_tensor=True)
            shadow_matte = shadow_matte.to(device)

            combined_input = torch.cat([input_img, shadow_matte], dim=1)  # [B, 4, H, W]

            output_img = model(combined_input)

            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)

            psnr = compute_psnr(output_img, target_img)
            ssim = compute_ssim(output_img, target_img)
            rmse = compute_rmse(output_img, target_img)

            total_psnr += psnr
            total_ssim += ssim
            total_rmse += rmse

            if args.save_outputs:
                for j in range(output_img.size(0)):
                    save_output_image(output_img[j:j+1], img_name[j], args.output_dir)

    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    avg_rmse = total_rmse / len(test_loader)
    avg_time = sum(processing_times) / len(processing_times)

    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"Average PSNR: {avg_psnr:.2f}dB")
    logger.info(f"Average SSIM: {avg_ssim:.4f}")
    logger.info(f"Average RMSE: {avg_rmse:.4f}")
    logger.info(f"Average Processing Time: {avg_time:.4f}s per image")
    logger.info("=" * 50)

def save_output_image(tensor, filename, output_dir):
    tensor = tensor.cpu()

    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = torch.clamp(tensor, 0, 1)

    tensor = tensor.permute(1, 2, 0).numpy()
    img = Image.fromarray((tensor * 255).astype(np.uint8))

    output_path = os.path.join(output_dir, filename)
    img.save(output_path)

if __name__ == "__main__":
    main()