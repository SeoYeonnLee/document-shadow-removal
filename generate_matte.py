'''
Construct shadow matte dataset
'''

import os
import cv2
import numpy as np

input_dir = '../data/Kligler/train/input'
target_dir = '../data/Kligler/train/target'
output_dir = '../data/Kligler/train/matte'

os.makedirs(output_dir, exist_ok=True)

file_list = sorted(os.listdir(input_dir))

for filename in file_list:
    input_path = os.path.join(input_dir, filename)
    target_path = os.path.join(target_dir, filename)
    output_path = os.path.join(output_dir, filename)

    shadow_img = cv2.imread(input_path)
    shadow_free_img = cv2.imread(target_path)

    shadow_lab = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    shadow_free_lab = cv2.cvtColor(shadow_free_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    L_shadow = shadow_lab[:, :, 0]
    L_free = shadow_free_lab[:, :, 0]

    shadow_matte = 1 - (L_shadow / (L_free + 1e-6))
    shadow_matte = np.clip(shadow_matte, 0, 1)

    matte_img = (shadow_matte * 255).astype(np.uint8)
    cv2.imwrite(output_path, matte_img)

    print(f"Saved matte: {output_path}")
