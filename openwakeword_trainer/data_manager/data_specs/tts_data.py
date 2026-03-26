from dataclasses import dataclass
from pathlib import Path

from .data_spec import DataSpec


@dataclass
class TTSData(DataSpec):

    neg_test: set
    neg_train: str
    pos_test: str
    pos_train: str

    def __init__ (self, tts_path: Path):
        DataSpec.__init__(
            self,
            final_path = tts_path,
            recreates = [
                tts_path / 'neg_test',
                tts_path / 'neg_train',
                tts_path / 'pos_test',
                tts_path / 'pos_train'
            ]
        )
        self.neg_test = str(tts_path / 'neg_test')
        self.neg_train = str(tts_path / 'neg_train')
        self.pos_test = str(tts_path / 'pos_test')
        self.pos_train = str(tts_path / 'pos_train')