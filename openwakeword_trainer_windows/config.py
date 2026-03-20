import yaml

from .data_manager import DataManager
from .resources import (
    AUDIOSET,
    FMA,
    MIT_RIRS,
    OWW_FEATURES,
    VALIDATION_FEATURES
)


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

        with open(DataManager.EX_CONF_PATH, 'r') as f:
            train = yaml.load(f.read(), yaml.Loader)

        train['model_name'] = self.model_name
        train['target_phrase'] = self.target_phrases
        train['custom_negative_phrases'] = self.negative_phrases
        train['n_samples'] = self.n_train
        train['n_samples_val'] = self.n_test
        train.pop('piper_sample_generator_path', None)
        train['output_dir'] = str(dm.training_path)
        train['rir_paths'] = [str(MIT_RIRS.path(dm.wav_path))]
        train['background_paths'] = [
            str(AUDIOSET.path(dm.wav_path)),
            str(FMA.path(dm.wav_path))
        ]
        train['background_paths_duplication_rate'] = [1, 1]
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