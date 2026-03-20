from dataclasses import dataclass
from pathlib import Path
import requests
import shutil

from ..logger import Logger

from .resource import Resource


@dataclass
class ModelResource(Resource):

    ### METHODS ###
    def download (self, local_path: Path):
        if self.is_downloaded(local_path):
            Logger.log(f'✅ {self.name} is already downloaded')
        else:
            dest = self.path(local_path)
            Logger.log(f'🔄 downloading {self.name}...')
            try:
                with requests.get(self.remote + self.name, stream=True) as r:
                    r.raise_for_status()
                    with open(str(dest), 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                Logger.log(f'✅ {self.name} successfully downloaded')
            except:
                Logger.log(f'❌ failed to download {self.name}')
                raise RuntimeError()

    def is_downloaded (self, local_path: Path) -> bool:
        return self.path(local_path).exists()
    
    def is_unpacked (self, local_path: Path) -> bool:
        return self.path(local_path).exists()
    
    def unpack (self, data_path: Path, dest_path: Path):
        if self.is_unpacked(dest_path):
            Logger.log(f'✅ {self.name} is already unpacked')
        else:
            Logger.log(f'📦 unpacking {self.name}...')
            if not dest_path.exists():
                Logger.log(f'❌ failed to unpack {self.name}')
                raise RuntimeError()
            src = self.path(data_path)
            dest = self.path(dest_path)
            shutil.copy(src, dest)
            Logger.log(f'✅ {self.name} successfully unpacked')


DEEP_PHONEMIZER = ModelResource(
    name = 'en_us_cmudict_forward.pt',
    remote = 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/'
)

EMBEDDING_MODEL = ModelResource(
    name = 'embedding_model.onnx',
    remote = 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/'
)

MELSPECTROGRAM_MODEL = ModelResource(
    name = 'melspectrogram.onnx',
    remote = 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/'
)

PIPER_TTS_JSON = ModelResource(
    name = 'en_US-libritts-high.onnx.json',
    remote = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/'
)

PIPER_TTS_ONNX = ModelResource(
    name = 'en_US-libritts-high.onnx',
    remote = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/'
)

SILERO_VAD_MODEL = ModelResource(
    name = 'silero_vad.onnx',
    remote = 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/'
)

MODEL_RESOURCES = [
    DEEP_PHONEMIZER,
    EMBEDDING_MODEL,
    MELSPECTROGRAM_MODEL,
    PIPER_TTS_JSON,
    PIPER_TTS_ONNX,
    SILERO_VAD_MODEL
]