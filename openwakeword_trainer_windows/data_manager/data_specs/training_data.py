from dataclasses import dataclass
from pathlib import Path

from .data_spec import DataSpec


@dataclass
class TrainingData(DataSpec):

    neg_test: str
    neg_train: str
    pos_test: str
    pos_train: str

    def __init__ (self, training_path: Path):
        DataSpec.__init__(
            self,
            final_path = [training_path],
            recreates = [
                training_path / 'neg_test',
                training_path / 'neg_train',
                training_path / 'pos_test',
                training_path / 'pos_train'
            ]
        )
        self.neg_test = str(training_path / 'neg_test')
        self.neg_train = str(training_path / 'neg_train')
        self.pos_test = str(training_path / 'pos_test')
        self.pos_train = str(training_path / 'pos_train')