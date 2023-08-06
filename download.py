from audiocraft.models import MusicGen
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
    HF_MODEL_CHECKPOINTS_MAP,
)
from audiocraft.data.audio import audio_write
import os

MODEL_PATH = "/home/pablo/cog-musicgen-runpod/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH

model_id = "facebook/musicgen-large"

name = next(
            (key for key, val in HF_MODEL_CHECKPOINTS_MAP.items() if val == model_id),
            None,
        )
lm = load_lm_model(name, device='cuda', cache_dir=MODEL_PATH)