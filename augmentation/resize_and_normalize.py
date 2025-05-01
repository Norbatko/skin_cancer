import os
from PIL import Image, ImageOps
from tqdm import tqdm

TARGET_SIZE = (1024, 1024)
INPUT_PATH = r"C:\Users\adamg\Desktop\ai_healthcare\images_malignant"
OUTPUT_PATH = r"C:\Users\adamg\Desktop\ai_healthcare\resized_and_filtered"

def pad_to_square(image):
    """Pad image to square using black bars."""
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    new_image = Image.new("RGB", (size, size), (0, 0, 0))  # black background
    new_image.paste(image, ((size - width) // 2, (size - height) // 2))
    return new_image

def normalize_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    total = 0
    kept = 0
    deleted = 0

    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc="Processing images"):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            input_path = os.path.join(root, file)
            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")  # Ensure consistent format
                    width, height = img.size
                    total += 1

                    if width >= TARGET_SIZE[0] and height >= TARGET_SIZE[1]:
                        # Pad to square and resize
                        square_img = pad_to_square(img)
                        resized_img = square_img.resize(TARGET_SIZE, Image.LANCZOS)

                        output_path = os.path.join(output_dir, file)
                        resized_img.save(output_path)
                        kept += 1
                    else:
                        deleted += 1

            except Exception as e:
                print(f"Error processing {file}: {e}")
                deleted += 1

    print(f"\nProcessed: {total}")
    print(f"Saved: {kept}")
    print(f"Discarded (too small or error): {deleted}")

if __name__ == "__main__":
    normalize_images(INPUT_PATH, OUTPUT_PATH)
