import os.path
from typing import Optional
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import requests
from tqdm import tqdm

MODELS_URL = {
    "appearance_feature_extractor": "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/appearance_feature_extractor.safetensors",
    "motion_extractor": "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/motion_extractor.safetensors",
    "warping_module": "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/warping_module.safetensors",
    "spade_generator": "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/spade_generator.safetensors",
    "stitching_retargeting_module ": "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/stitching_retargeting_module.safetensors",
    "face_yolov8n": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
}


def download_model(
    file_path: str,
    url: str,
) -> Optional[str]:
    if os.path.exists(file_path):
        return None
    try:
        print(f'{file_path} is not detected. Downloading model...')
        download_url_to_file(url, file_path, progress=True)

    except requests.exceptions.RequestException as e:
        print(
            f"Model download has failed. Please download manually from: {url}\n"
            f"and place it in {file_path}"
        )
        raise e
    except Exception as e:
        print('An unexpected error occurred during model download.')
        raise e
