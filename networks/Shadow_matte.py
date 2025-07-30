import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# shadow matte generator
class ShadowMattePredictor:
    def __init__(self, model_path, device=None, img_size=256):

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        self.net = UNet(n_channels=3, n_classes=1, bilinear=False)
        self.net.to(device=self.device)
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        if 'mask_values' in state_dict:
            state_dict.pop('mask_values')
        
        self.net.load_state_dict(state_dict)
        self.net.eval()
    
    @staticmethod
    def preprocess(pil_img, img_size):
        pil_img = pil_img.resize((img_size, img_size), resample=Image.Resampling.BICUBIC)
        img = np.asarray(pil_img)
        
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))
        
        if (img > 1).any():
            img = img / 255.0
            
        return img
    
    def predict(self, input_data=None, is_tensor=True):
        if is_tensor:
            img = input_data
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(device=self.device, dtype=torch.float32)
        else:
            pil_img = Image.open(input_data)
            img = self.preprocess(pil_img, self.img_size)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img).cpu()
            mask = torch.sigmoid(output)

        if is_tensor and input_data.dim() == 4:
            predicted_matte = mask
        else:
            predicted_matte = mask[0].unsqueeze(0)
        
        return predicted_matte
    
    def save_matte(self, matte, output_path):
        matte_img = (matte * 255).astype(np.uint8)
        result = Image.fromarray(matte_img)
        result.save(output_path)

if __name__ == "__main__":
    matte_predictor = ShadowMattePredictor(model_path="checkpoints/UNet/image_256/epoch180.pth", img_size=256)

    # Using image path
    matte = matte_predictor.predict(input_data="test.jpg", is_tensor=False)

    # Using tensor input
    tensor_img = transforms.ToTensor()(Image.open("test.jpg"))
    batch_tensor = torch.stack([tensor_img, tensor_img])
    batch_mattes = matte_predictor.predict(input_data=batch_tensor, is_tensor=True)