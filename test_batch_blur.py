import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random

class BatchedBlurAugment:
    def __init__(self, blur_prob=0.8, min_kernel=3, max_kernel=15, device='cpu'):
        self.blur_prob = blur_prob
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.device = device
        
    def _get_motion_blur_kernel(self, kernel_size, angle):
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(kernel_size))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel_size, kernel_size))
        motion_blur_kernel = motion_blur_kernel / np.sum(motion_blur_kernel)
        return torch.from_numpy(motion_blur_kernel).float()

    def _get_gaussian_blur_kernel(self, kernel_size):
        # 1D gaussian
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        # 2D gaussian
        gauss_2d = gauss_1d[:, None] @ gauss_1d[None, :]
        return gauss_2d

    def __call__(self, images_tensor):
        """
        images_tensor: (B, C, H, W) on self.device
        """
        B, C, H, W = images_tensor.shape
        blurred_images = images_tensor.clone()
        
        for i in range(B):
            if random.random() > self.blur_prob:
                continue
                
            kernel_size = random.randint(self.min_kernel, self.max_kernel)
            if kernel_size % 2 == 0: kernel_size += 1
                
            if random.random() < 0.6:
                angle = random.randint(0, 360)
                kernel = self._get_motion_blur_kernel(kernel_size, angle).to(self.device)
            else:
                kernel = self._get_gaussian_blur_kernel(kernel_size).to(self.device)
                
            kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, kernel_size, kernel_size)
            padding = kernel_size // 2
            
            # Apply to single image i
            img = images_tensor[i:i+1] # (1, C, H, W)
            blurred = F.conv2d(img, kernel, padding=padding, groups=C)
            blurred_images[i] = blurred.squeeze(0)
            
        return blurred_images

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    aug = BatchedBlurAugment(blur_prob=1.0, device=device)
    x = torch.rand(4, 3, 256, 256).to(device)
    out = aug(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print("Success, fast GPU-enabled blurring ran fine.")
