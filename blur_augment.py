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
