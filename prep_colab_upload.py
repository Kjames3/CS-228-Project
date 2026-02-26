import os
import zipfile
from pathlib import Path

def zip_custom_datasets(source_dir, output_zip):
    print(f"Creating {output_zip}...")
    print("Zipping custom datasets (skipping massive COCO files to save upload time)...")
    
    source_path = Path(source_dir)
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_path):
            current_path = Path(root)
            
            # Skip coco datasets, environments, and cache
            if "coco" in current_path.name.lower() or "coco" in str(current_path).lower():
                continue
            if "__pycache__" in current_path.name or ".venv" in str(current_path):
                continue
                
            for file in files:
                # Skip massive zips if present just in case
                if "coco" in file.lower() and file.endswith(".zip"):
                    continue
                    
                file_path = current_path / file
                arcname = file_path.relative_to(source_path.parent)
                
                zipf.write(file_path, arcname)
                
    print(f"✅ Created {output_zip} successfully!")
    print(f"Size: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")
    print("\nNext step: Upload this ZIP file to your Google Drive in a folder named 'CS-228-Project'")

if __name__ == "__main__":
    datasets_dir = r"c:\Users\besto\OneDrive\Documents\Viam Rover 2 projects\viam_projects\datasets"
    output_file = "custom_datasets_only.zip"
    
    if os.path.exists(datasets_dir):
        zip_custom_datasets(datasets_dir, output_file)
    else:
        # Fallback to local project datasets if the global one doesn't exist
        zip_custom_datasets("datasets", output_file)
