import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

# ===== Constants =====
IMG_SIZE = 352
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Load model =====
def load_model(model_path):
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ===== Inference function =====
def run_segmentation(model, image_path, output_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("❌ Image cannot be read")

    # Handle grayscale images
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Save original size
    h, w = img.shape[:2]

    # Preprocess
    x = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x = x / 255.0
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    # Inference
    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    # Threshold
    pred = (pred > THRESHOLD).astype("uint8") * 255

    # Resize back to original size
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    # Save mask
    cv2.imwrite(output_path, pred)
