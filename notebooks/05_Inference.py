# 05_Inference.py
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import pathlib

# Resolve project root (plant_detection/)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models" / "leafnet_v1"
IMG_SIZE = (224, 224)

# Load class names
class_names_path = PROJECT_ROOT / "models" / "class_names.json"
if not class_names_path.exists():
    raise FileNotFoundError(f"❌ class_names.json not found at {class_names_path}")

with open(class_names_path, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Load model (SavedModel)
model = tf.keras.models.load_model(str(MODELS_DIR / "saved_models"))

def preprocess_image_pil(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to model-ready numpy array."""
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)
    return arr

def predict_topk(pil_img: Image.Image, top_k=3):
    """Return top-k predictions as (class_name, probability)."""
    x = preprocess_image_pil(pil_img)
    preds = model.predict(x, verbose=0)[0]
    topk_idx = np.argsort(preds)[-top_k:][::-1]
    return [(class_names[i], float(preds[i])) for i in topk_idx]

# Example usage
if __name__ == "__main__":
    sample_path = PROJECT_ROOT / "data" / "raw" / "PlantVillage" / class_names[0]

    found = False
    for root, dirs, files in os.walk(sample_path):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, fn)
                img = Image.open(img_path)
                results = predict_topk(img, top_k=3)

                print(f"\nSample image: {img_path}")
                for rank, (cls, prob) in enumerate(results, 1):
                    print(f"  {rank}. {cls:40s}  {prob:.6f}")
                found = True
                break
        if found:
            break

    if not found:
        print("⚠️ No sample image found in the expected directory.")
