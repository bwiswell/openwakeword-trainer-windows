from dataclasses import dataclass
from pathlib import Path

from ..resources import (
    DEEP_PHONEMIZER,
    EMBEDDING_MODEL,
    MELSPEC_MODEL,
    PIPER_TTS_JSON,
    PIPER_TTS_ONNX,
    SILERO_VAD_MODEL
)

from .data_spec import DataSpec


@dataclass
class ModelData(DataSpec):

    deep_phonemizer: str
    embedding_model: str
    melspec_model: str
    piper_tts_model: str
    silero_vad_model: str

    def __init__ (self, resource_path: Path):
        DataSpec.__init__(
            self,
            final_path = resource_path,
            download_path = resource_path,
            ensures = [resource_path],
            resources = [
                DEEP_PHONEMIZER,
                EMBEDDING_MODEL,
                MELSPEC_MODEL,
                PIPER_TTS_JSON,
                PIPER_TTS_ONNX,
                SILERO_VAD_MODEL
            ]
        )
        self.deep_phonemizer = str(DEEP_PHONEMIZER.path(resource_path))
        self.embedding_model = str(EMBEDDING_MODEL.path(resource_path))
        self.melspec_model = str(MELSPEC_MODEL.path(resource_path))
        self.piper_tts_model = str(PIPER_TTS_ONNX.path(resource_path))
        self.silero_vad_model = str(SILERO_VAD_MODEL.path(resource_path))