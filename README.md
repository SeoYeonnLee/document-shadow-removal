# MatteViT: High-Frequency-Aware Document Shadow Removal With Shadow Matte Guidance

MatteViT is a transformer-based framework for document shadow removal that effectively preserves fine text details using luminance-based shadow matte guidance, high-frequency enhancement, and frequency-sensitive loss.

<img src="https://github.com/user-attachments/assets/9be03594-cc4b-46d3-93f4-4111f09c0dc3">

## Installation
```
pip install -r requirements.txt
```
<br>

## Dataset Preparation
Organize your dataset as follows:
```
data/
├── train/
│   ├── input/     # Training shadow images
│   ├── target/    # Training shadow-free images
│   └── matte/     # Training shadow mattes (generated)
└── test/
   ├── input/     # Test shadow images
   ├── target/    # Test shadow-free images
   └── matte/     # Test shadow mattes (generated)
```
<br>

## Generate Shadow Mattes
```
python generate_matte.py
```
<br>

## Training
1. Train Shadow Matte Generator
```
python train_unet.py
```
2. Train MatteViT
```
python train_vit.py \
    --training_path <TRAIN_DIR> \
    --shadow_matte_path <MATTE_GENERATOR_PATH>
```
<br>

## Testing
```
python test_vit.py \
    --model_path <VIT_MODEL_PATH> \
    --test_input_dir <TEST_INPUT_DIR> \
    --test_target_dir <TEST_TARGET_DIR> \
    --shadow_matte_path <MATTE_GENERATOR_PATH>
```
