import itertools as it
import os
from pathlib import Path
import shutil
from typing import Optional

from .logger import Logger
from .resources import (
    FEATURE_RESOURCES,
    GIT_RESOURCES,
    MODEL_RESOURCES,
    WAV_RESOURCES
)


class DataManager:

    CWD = Path(os.path.realpath(os.path.dirname(__file__)))
    PARENT = Path(os.path.realpath(CWD / '..'))
    DEFAULT_DATA_PATH = PARENT / 'data'
    DEFAULT_OUTPUT_PATH = PARENT / 'output'
    EX_CONF_PATH = PARENT / 'openwakeword' / 'examples' / 'custom_model.yml'
    MODEL_PATH = PARENT / 'openwakeword' / 'openwakeword' / 'resources' / 'models'
    SCRIPT_PATH = PARENT / 'openwakeword' / 'openwakeword' / 'train.py'
    SCRIPT_DATA_PATH = PARENT / 'openwakeword' / 'openwakeword' / 'data.py'
    SCRIPT_UTILS_PATH = PARENT / 'openwakeword' / 'openwakeword' / 'utils.py'

    def __init__ (
                self,
                model: str,
                data_dir: Optional[str] = str(DEFAULT_DATA_PATH),
                output_dir: Optional[str] = str(DEFAULT_OUTPUT_PATH)
            ):
        self.model = model
        self.data_path = Path(data_dir)
        self.output_path = Path(output_dir)

        self.cache_path = self.data_path / 'datasets' / 'cache'
        self.dataset_path = self.data_path / 'datasets'
        self.resource_path = self.data_path / 'resources'
        self.wav_path = self.data_path / 'wavs'

        self.config_path = DataManager.PARENT / 'configs' / f'{model}.yaml'

        self.record_path = self.data_path / 'recordings' / model
        self.record_pos_path = self.record_path / 'positive'
        self.record_neg_path = self.record_path / 'negative'

        self.train_path = self.data_path / 'training' / model
        self.train_conf_path = self.train_path / f'{model}.yaml'
        self.training_path = self.data_path / 'training'

        self.pos_train = self.train_path / 'positive_train'
        self.neg_train = self.train_path / 'negative_train'
        self.pos_test = self.train_path / 'positive_test'
        self.neg_test = self.train_path / 'negative_test'


    ### PROPERTIES ###
    @property
    def n_train_neg (self) -> int:
        return len(list(self.neg_train.glob('*.wav')))
    
    @property
    def n_train_pos (self) -> int:
        return len(list(self.pos_train.glob('*.wav')))

    @property
    def n_recorded_neg (self) -> int:
        return len(list(self.record_neg_path.glob('*.wav')))
    
    @property
    def n_recorded_pos (self) -> int:
        return len(list(self.record_pos_path.glob('*.wav')))


    ### METHODS ###
    def download (self):
        Logger.log('🚀 starting resource downloads...')
        try:
            for gr in GIT_RESOURCES:
                gr.download(DataManager.PARENT)
            for fr in FEATURE_RESOURCES:
                fr.download(self.resource_path)
            for mr in MODEL_RESOURCES:
                mr.download(self.resource_path)
            for wr in WAV_RESOURCES:
                wr.download(self.dataset_path)
        except Exception as e:
            Logger.log(f'❌ failed to download resources')
            raise e
        Logger.log('✨ all resources downloaded')

    def ensure_paths (self):
        Logger.log('🚀 (re)creating resource paths...')
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.record_pos_path.mkdir(parents=True, exist_ok=True)
        self.record_neg_path.mkdir(parents=True, exist_ok=True)
        self.resource_path.mkdir(parents=True, exist_ok=True)
        if self.train_path.exists(): shutil.rmtree(self.train_path)
        self.train_path.mkdir(parents=True)
        self.pos_train.mkdir()
        self.neg_train.mkdir()
        self.pos_test.mkdir()
        self.neg_test.mkdir()
        self.wav_path.mkdir(parents=True, exist_ok=True)
        Logger.log('✨ all resources paths created')

    def export (self):
        Logger.log('🚀 exporting models...')
        onnx_in = self.training_path / f'{self.model}.onnx'
        #tflite_in = self.training_path / f'{self.model}.tflite'
        if not onnx_in.exists():# or not tflite_in.exists():
            Logger.log(f'❌ models unvailable for export')
            raise RuntimeError()
        onnx_out = self.output_path / f'{self.model}.onnx'
        #tflite_out = self.output_path / f'{self.model}.tflite'
        if onnx_out.exists(): os.remove(onnx_out)
        #if tflite_out.exists(): os.remove(tflite_out)
        os.rename(onnx_in, onnx_out)
        #os.rename(tflite_in, tflite_out)
        Logger.log('✨ all models exported')

    def unpack (self):
        Logger.log('🚀 starting resource unpacking...')
        try:
            for gr in GIT_RESOURCES:
                gr.unpack(DataManager.PARENT, DataManager.PARENT)
            for fr in FEATURE_RESOURCES:
                fr.unpack(self.resource_path, self.resource_path)
            for mr in MODEL_RESOURCES:
                print(DataManager.MODEL_PATH)
                mr.unpack(self.resource_path, DataManager.MODEL_PATH)
            for wr in WAV_RESOURCES:
                wr.unpack(self.dataset_path, self.wav_path)
        except Exception as e:
            Logger.log(f'❌ failed to unpack resources')
            raise e
        Logger.log('✨ all resources unpacked')