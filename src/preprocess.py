import os
import numpy as np
import pydicom

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_image(path):
    # try DICOM first
    if path.lower().endswith(".dcm"):
        ds = pydicom.dcmread(path)
        image = ds.pixel_array
    else:
        # fallback for jpg/png (for testing)
        from PIL import Image
        img = Image.open(path).convert("L").resize((256, 256))
        image = np.array(img)

    return image

def normalize(image):
    image = image.astype("float32")
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

def process_file(file_path):
    image = load_image(file_path)
    image = normalize(image)

    filename = os.path.basename(file_path)
    name, _ = os.path.splitext(filename)

    output_path = os.path.join(PROCESSED_DIR, name + ".npy")
    np.save(output_path, image)

    return output_path