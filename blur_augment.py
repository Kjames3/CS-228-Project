import cv2
import numpy as np
import random

class BlurAugment:
    """
    Handles synthetic blur generation to simulate motion and vibration
    artifacts observed on the Viam Rover.
    """
    def __init__(self, blur_prob=0.5, min_kernel=3, max_kernel=15):
        self.blur_prob = blur_prob
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel

    def apply_motion_blur(self, image, kernel_size=None, angle=None):
        """
        Applies motion blur to an image.
        """
        if kernel_size is None:
            kernel_size = random.randint(self.min_kernel, self.max_kernel)
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        if angle is None:
            angle = random.randint(0, 360)

        # Create motion blur kernel
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(kernel_size))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel_size, kernel_size))
        motion_blur_kernel = motion_blur_kernel / kernel_size

        # Apply filter
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        return blurred

    def apply_gaussian_blur(self, image, kernel_size=None):
        """
        Applies Gaussian blur to an image (simulating out-of-focus).
        """
        if kernel_size is None:
            kernel_size = random.randint(self.min_kernel, self.max_kernel)
            
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def __call__(self, image):
        """
        Apply random blur to the image with probability self.blur_prob.
        Expects image in BGR (OpenCV) or RGB format.
        """
        if random.random() > self.blur_prob:
            return image

        # 50/50 chance of motion blur vs gaussian blur (or simulate mixed vibration)
        if random.random() < 0.6:
            # Motion blur is more likely for a moving rover
            return self.apply_motion_blur(image)
        else:
            return self.apply_gaussian_blur(image)

if __name__ == "__main__":
    # Test execution
    print("Testing BlurAugment...")
    aug = BlurAugment(blur_prob=1.0)
    
    # Create a dummy image (checkerboard)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[::20, ::20] = 255
    
    blurred = aug(img)
    cv2.imwrite("project/debug_samples/test_blur_output.jpg", blurred)
    print("Saved test output to project/debug_samples/test_blur_output.jpg")

class BatchedBlurAugment:
    """
    PyTorch Batched GPU implementation of BlurAugment to optimize training speeds.
    Operates entirely on (B, C, H, W) tensors.
    """
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
        import torch
        return torch.from_numpy(motion_blur_kernel).float()

    def _get_gaussian_blur_kernel(self, kernel_size):
        import torch
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
        import torch
        import torch.nn.functional as F
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

