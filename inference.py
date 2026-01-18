import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

IMG_SIZE = 352
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_REPO_ID = "khaled314/capsule-segmentation-model"
HF_MODEL_FILE = "best_model.pth"


def download_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILE
    )
    return model_path

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


def run_segmentation(model, image_path, output_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("❌ Image cannot be read")

    # Handle grayscale images
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w = img.shape[:2]


    x = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x = x / 255.0
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    pred = (pred > THRESHOLD).astype("uint8") * 255

    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(output_path, pred)
