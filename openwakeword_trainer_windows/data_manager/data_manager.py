import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional

from ..logger import Logger

from .data_specs import (
    FeatureData,
    ModelData,
    OutputData,
    RecordingData,
    TrainingData,
    TTSData,
    WavData
)


class DataManager:

    CWD = Path(os.path.realpath(os.path.dirname(__file__)))
    PARENT = Path(os.path.realpath(CWD / '..'))
    DEFAULT_DATA_PATH = PARENT / 'data'
    DEFAULT_OUTPUT_PATH = PARENT / 'outputs'

    def __init__ (
                self,
                model: str,
                data_dir: Optional[str] = str(DEFAULT_DATA_PATH),
                output_dir: Optional[str] = str(DEFAULT_OUTPUT_PATH)
            ):
        self.model = model
        self.data_path = Path(data_dir)

        self.cache_path = self.data_path / 'datasets' / 'cache'
        self.dataset_path = self.data_path / 'datasets'
        self.resource_path = self.data_path / 'resources'
        self.wav_path = self.data_path / 'wavs'

        self.config_path = DataManager.PARENT / 'configs' / f'{model}.yaml'
        self.export_path = self.data_path / 'exports' / model
        self.feature_path = self.data_path / 'features' / model
        self.output_path = Path(output_dir) / model
        self.recording_path = self.data_path / 'recordings' / model
        self.training_path = self.data_path / 'training' / model
        self.tts_path = self.data_path / 'tts' / model

        self.features = FeatureData(self.resource_path, self.feature_path)
        self.models = ModelData(self.resource_path)
        self.outputs = OutputData(self.export_path, self.output_path, model)
        self.recordings = RecordingData(self.recording_path)
        self.training = TrainingData(self.training_path)
        self.tts = TTSData(self.tts_path)
        self.wavs = WavData(self.dataset_path, self.wav_path)


    ### METHODS ###
    def download (self):
        Logger.log('🚀 starting resource downloads...')
        try:
            self.features.download()
            self.models.download()
            self.wavs.download()
        except Exception as e:
            Logger.log(f'❌ failed to download resources')
            raise e
        Logger.log('✨ all resources downloaded')


    def ensure_paths (self):
        Logger.log('🚀 (re)creating resource paths...')
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.features.ensure()
        self.models.ensure()
        self.outputs.ensure()
        self.recordings.ensure()
        self.training.ensure()
        self.tts.ensure()
        self.wavs.ensure()
        Logger.log('✨ all resources paths created')


    def unpack (self):
        Logger.log('🚀 starting resource unpacking...')
        try:
            self.wavs.unpack()
        except Exception as e:
            Logger.log(f'❌ failed to unpack resources')
            raise e
        Logger.log('✨ all resources unpacked')