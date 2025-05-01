import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm


TARGET_SIZE = (1024, 1024)
INPUT_PATH = r"C:\Users\adamg\Desktop\ai_healthcare\images_malignant"
OUTPUT_PATH = r"C:\Users\adamg\Desktop\ai_healthcare\resized_and_filtered"

def pad_to_square(image):
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    new_image = Image.new("RGB", (size, size), (0, 0, 0))
    new_image.paste(image, ((size - width) // 2, (size - height) // 2))
    return new_image

def process_image(file_path, output_dir):
    try:
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            if width >= TARGET_SIZE[0] and height >= TARGET_SIZE[1]:
                square_img = pad_to_square(img)
                resized_img = square_img.resize(TARGET_SIZE, Image.LANCZOS)
                output_path = os.path.join(output_dir, os.path.basename(file_path))
                resized_img.save(output_path)
                return "saved"
            else:
                return "skipped"
    except Exception:
        return "error"

def normalize_images_multithreaded(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = list(input_dir.rglob("*"))
    image_files = [f for f in all_files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    kept = 0
    deleted = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_image, f, output_dir) for f in image_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result == "saved":
                kept += 1
            else:
                deleted += 1

    print(f"\\nTotal processed: {{len(image_files)}}")
    print(f"Saved: {{kept}}")
    print(f"Skipped or errors: {{deleted}}")

if __name__ == "__main__":
    normalize_images_multithreaded(INPUT_PATH, OUTPUT_PATH)