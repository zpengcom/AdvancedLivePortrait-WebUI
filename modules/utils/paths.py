import functools
import os


PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT_DIR, "outputs")
MODEL_CONFIG = os.path.join(PROJECT_ROOT_DIR, "modules", "config", "models.yaml")
MODEL_PATHS = {
    "appearance_feature_extractor": os.path.join(MODELS_DIR, "appearance_feature_extractor.safetensors"),
    "motion_extractor": os.path.join(MODELS_DIR, "motion_extractor.safetensors"),
    "warping_module": os.path.join(MODELS_DIR, "warping_module.safetensors"),
    "spade_generator": os.path.join(MODELS_DIR, "spade_generator.safetensors"),
    "stitching_retargeting_module": os.path.join(MODELS_DIR, "stitching_retargeting_module.safetensors")
}


@functools.lru_cache
def init_dirs():
    for dir_path in [
        MODELS_DIR,
        OUTPUTS_DIR
    ]:
        os.makedirs(dir_path, exist_ok=True)


init_dirs()
