import requests
from tqdm import tqdm

MODELS = {
    "appearance_feature_extractor": "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/appearance_feature_extractor.safetensors",
    "motion_extractor": "https://huggingface.co/Kijai/LivePortrait_safetensors/blob/main/motion_extractor.safetensors",
    "warping_module": "https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main/warping_module.safetensors",
    "spade_generator ": "https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main/spade_generator.safetensors ",
    "stitching_retargeting_module ": "https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main/stitching_retargeting_module.safetensors",
}


def download_model(file_path, model_url):
    print('Downloading model...')
    response = requests.get(model_url, stream=True)
    try:
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            # tqdm will display a progress bar
            with open(file_path, 'wb') as file, tqdm(
                    desc='Downloading',
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    file.write(data)

    except requests.exceptions.RequestException as e:
        print(
            f"Model Download has failed. Download manually from: {model_url}"
            f"And place in {file_path}"
        )
        raise
    except Exception as e:
        print(f'An unexpected error occurred during model download')
        raise

