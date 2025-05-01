from collections import Counter
from PIL import Image
import os

FOLDER_PATH = r"C:\Users\adamg\Desktop\ai_healthcare\resized_and_filtered"

def collect_resolutions(folder_path):
    resolutions = Counter()
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                try:
                    with Image.open(os.path.join(root, file)) as img:
                        resolutions[img.size] += 1
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    return resolutions

if __name__ == "__main__":
    res_counts = collect_resolutions(FOLDER_PATH)
    
    for res, count in sorted(res_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"Resolution {res}: {count} images")
