from dataclasses import dataclass
import os
from pathlib import Path

from .data_spec import DataSpec


@dataclass
class RecordingData(DataSpec):

    positive: str
    negative: str

    def __init__ (self, recording_path: Path):
        DataSpec.__init__(
            self,
            final_path = recording_path,
            ensures = [
                recording_path / 'positive',
                recording_path / 'negative'
            ]
        )
        self.positive = str(recording_path / 'positive')
        self.negative = str(recording_path / 'negative')


    ### PROPERTIES ###
    @property
    def n_negative (self) -> int:
        return len(list(os.scandir(self.negative)))

    @property
    def n_positive (self) -> int:
        return len(list(os.scandir(self.positive)))