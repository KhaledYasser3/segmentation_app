import torch
import numpy as np
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
from PIL import Image

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 352
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔴 غيّر ده لاسم الريبو بتاعك
HF_REPO_ID = "khaled314/capsule-segmentation-model"
HF_MODEL_FILE = "best_model.pth"


# ===============================
# DOWNLOAD MODEL
# ===============================
def download_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILE
    )
    return model_path


# ===============================
# LOAD MODEL
# ===============================
def load_model():
    model_path = download_model()

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


# ===============================
# RUN SEGMENTATION
# ===============================
def run_segmentation(model, image_path, output_path):
    img = Image.open(image_path).convert("RGB")

    # Original size
    w, h = img.size

    # Preprocess
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img_resized) / 255.0
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    # Inference
    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    # Threshold
    pred = (pred > THRESHOLD).astype("uint8") * 255

    # Resize back
    pred_img = Image.fromarray(pred).resize((w, h), resample=Image.NEAREST)

    # Save
    pred_img.save(output_path)
