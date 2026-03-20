from dataclasses import dataclass
from pathlib import Path
import shutil
from urllib import request

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
                request.urlretrieve(self.remote + self.name, str(dest))
                Logger.log(f'✅ {self.name} successfully downloaded')
            except:
                Logger.log(f'❌ failed to download {self.name}')
                raise Exception()

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


EMBEDDING_MODEL = ModelResource(
    name = 'embedding_model.onnx',
    remote = 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/'
)

MELSPECTROGRAM_MODEL = ModelResource(
    name = 'melspectrogram.onnx',
    remote = 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/'
)

SILERO_VAD_MODEL = ModelResource(
    name = 'silero_vad.onnx',
    remote = 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/'
)

MODEL_RESOURCES = [EMBEDDING_MODEL, MELSPECTROGRAM_MODEL, SILERO_VAD_MODEL]