####
# Author: Adam Gabrys
# Simple script to flip and rotate images in a directory multithreaded

import os
import cv2
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# ===== ENUMS =====
class Rotation(Enum):
    NONE = 0
    DEG_90 = 90
    DEG_180 = 180
    DEG_270 = 270

# ===== CONFIG =====
FLIP = True
ROTATE = Rotation.DEG_180
IMAGE_DIR = 'images_malignant'
OUTPUT_PATH = 'imaged_augmented'
# ===================

os.makedirs(OUTPUT_PATH, exist_ok=True)

def process_image(filename):
    input_path = os.path.join(IMAGE_DIR, filename)
    name, ext = os.path.splitext(filename)

    try:
        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: Couldn't read {filename}")
            return

        suffix_parts = []

        if FLIP:
            img = cv2.flip(img, 1)
            suffix_parts.append("flipped")

        if ROTATE != Rotation.NONE:
            if ROTATE == Rotation.DEG_90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif ROTATE == Rotation.DEG_180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif ROTATE == Rotation.DEG_270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            suffix_parts.append(f"rotated_{ROTATE.value}")

        if not suffix_parts:
            return  # Skip saving if nothing changed

        suffix = "_" + "_".join(suffix_parts)
        output_filename = f"{name}{suffix}{ext}"
        output_path = os.path.join(OUTPUT_PATH, output_filename)

        cv2.imwrite(output_path, img)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

def main():
    files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    with ThreadPoolExecutor() as executor:
        executor.map(process_image, files)

if __name__ == "__main__":
    main()
