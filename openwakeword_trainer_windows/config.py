from pathlib import Path
import yaml

from .data_manager import DataManager


class Config:

    def __init__ (self, dm: DataManager):
        with open(dm.config_path, 'r') as f:
            user = yaml.load(f.read(), yaml.Loader)

        self.model_name: str = user['model_name']
        self.target_phrases: list[str] = user['target_phrases']
        self.negative_phrases: list[str] = user['negative_phrases']
        self.n_train: int = user['training_samples']
        self.n_test: int = user['testing_samples']
        self.augmentations: int = user['augmentation_rounds']
        self.layer_size = user['layer_size']
        self.steps = user['steps']
        self.target_fp = user['target_fp']

        self.augmentation_batch = 16
        self.background_paths = [dm.wavs.audioset, dm.wavs.fma]
        self.rir_path = dm.wavs.rirs

        self.feature_data = {
            'ACAV100M': (1024, dm.features.acav),
            'negative': (50, dm.features.neg_train),
            'positive': (50, dm.features.pos_train),
        }

        '''
        train['false_positive_validation_data_path'] = str(
            VALIDATION_FEATURES.path(dm.resource_path)
        )
        train['augmentation_rounds'] = self.augmentations
        train['feature_data_files'] = {
            'ACAV100M_sample': str(OWW_FEATURES.path(dm.resource_path))
        }
        train['batch_n_per_class'] = {
            "ACAV100M_sample": 1024,
            "adversarial_negative": 50,
            "positive": 50
        }
        train['layer_size'] = self.layer_size
        train['steps'] = self.steps
        train['target_false_positives_per_hour'] = self.target_fp

        with open(dm.train_conf_path, 'w') as f:
            yaml.dump(train, f)
        '''