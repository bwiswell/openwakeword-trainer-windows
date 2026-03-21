from dataclasses import dataclass
from pathlib import Path

from ..resources import (
    AUDIOSET,
    FMA,
    MIT_RIRS
)

from .data_spec import DataSpec


@dataclass
class WavData(DataSpec):

    audioset: str
    fma: str
    rirs: str

    def __init__ (self, dataset_path: Path, wav_path: Path):
        DataSpec.__init__(
            self,
            final_path = wav_path,
            download_path = dataset_path,
            ensures = [dataset_path, wav_path],
            resources = [AUDIOSET, FMA, MIT_RIRS]
        )
        self.audioset = AUDIOSET.path(wav_path)
        self.fma = FMA.path(wav_path)
        self.rirs = MIT_RIRS.path(wav_path)