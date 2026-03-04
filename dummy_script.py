import os
import random
from ultralytics import YOLO

def main():
    # 1. Load the model
    model_path = "runs/teachers/yolov8_teacher/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # 2. View Model Architecture
    print("\n--- Model Architecture ---")
    model.info()
    
    # 3. View Model Accuracy
    # Running validation against the validation set to calculate current metrics
    print("\n--- Model Accuracy ---")
    print("Running validation to get accuracy metrics (this might take a moment)...")
    # Ultralytics .val() uses the dataset yaml referenced in the model's training args automatically
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    
    # 4. Test on 10 random images
    print("\n--- Testing on 10 Random Images ---")
    images_dir = "datasets/model_training_data/test/images/"
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return
        
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        print("No images found in the test directory.")
        return
        
    # Select 10 random images
    selected_images = random.sample(all_images, min(10, len(all_images)))
    selected_paths = [os.path.join(images_dir, img) for img in selected_images]
    
    print(f"Running inference on {len(selected_paths)} images...")
    # Run inference and save the results
    results = model(selected_paths, save=True)
    
    print("\nDone! Check the 'runs/detect/predict' folder (or similar predict folder) for the saved output images.")

if __name__ == '__main__':
    main()