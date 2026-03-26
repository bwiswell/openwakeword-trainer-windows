from dataclasses import dataclass
from pathlib import Path

from ..resources import OWW_FEATURES, VALIDATION_FEATURES

from .data_spec import DataSpec


@dataclass
class FeatureData(DataSpec):

    acav: str
    neg_test: str
    neg_train: str
    pos_test: str
    pos_train: str
    validation: str

    def __init__ (self, resource_path: Path, feature_path: Path):
        DataSpec.__init__(
            self,
            final_path = feature_path,
            download_path = resource_path,
            ensures = [resource_path],
            recreates = [feature_path],
            resources = [OWW_FEATURES, VALIDATION_FEATURES]
        )
        self.acav = str(OWW_FEATURES.path(resource_path))
        self.neg_test = str(feature_path / 'neg_test.npy')
        self.neg_train = str(feature_path / 'neg_train.npy')
        self.pos_test = str(feature_path / 'pos_test.npy')
        self.pos_train = str(feature_path / 'pos_train.npy')
        self.validation = str(VALIDATION_FEATURES.path(resource_path))