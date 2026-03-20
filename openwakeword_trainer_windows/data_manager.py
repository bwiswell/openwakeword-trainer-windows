import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional

from .logger import Logger
from .resources import (
    FEATURE_RESOURCES,
    MODEL_RESOURCES,
    WAV_RESOURCES
)


class DataManager:

    CWD = Path(os.path.realpath(os.path.dirname(__file__)))
    PARENT = Path(os.path.realpath(CWD / '..'))
    DEFAULT_DATA_PATH = PARENT / 'data'
    DEFAULT_OUTPUT_PATH = PARENT / 'outputs'
    EX_CONF_PATH = PARENT / 'openwakeword' / 'examples' / 'custom_model.yml'
    MODEL_PATH = PARENT / 'models'

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
        DataManager.MODEL_PATH.mkdir(exist_ok=True)
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
        if not onnx_in.exists():
            Logger.log(f'❌ no Onnx model found')
            raise RuntimeError()
        stats_in = self.training_path / f'{self.model}.json'
        if not stats_in.exists():
            Logger.log(f'❌ no stats file found')
            raise RuntimeError()
        Logger.log('🔄 converting Onnx model to TFLite...')
        subprocess.run([
            sys.executable, '-m', 'onnx2tf',
            '-i', str(onnx_in),
            '-o', str(self.train_path),
            '-tb', 'flatbuffer_direct'
        ], check=True)
        tflite_in = self.train_path / f'{self.model}_float32.tflite'
        if not tflite_in.exists():
            Logger.log(f'❌ TFLite conversion failed')
            raise RuntimeError()
        onnx_out = self.output_path / f'{self.model}.onnx'
        tflite_out = self.output_path / f'{self.model}.tflite'
        stats_out = self.output_path / f'{self.model}.json'
        if onnx_out.exists(): os.remove(onnx_out)
        if tflite_out.exists(): os.remove(tflite_out)
        if stats_out.exists(): os.remove(stats_out)
        os.rename(onnx_in, onnx_out)
        os.rename(tflite_in, tflite_out)
        os.rename(stats_in, stats_out)
        Logger.log('✨ all models exported')

    def unpack (self):
        Logger.log('🚀 starting resource unpacking...')
        try:
            for fr in FEATURE_RESOURCES:
                fr.unpack(self.resource_path, self.resource_path)
            for mr in MODEL_RESOURCES:
                mr.unpack(self.resource_path, DataManager.MODEL_PATH)
            for wr in WAV_RESOURCES:
                wr.unpack(self.dataset_path, self.wav_path)
        except Exception as e:
            Logger.log(f'❌ failed to unpack resources')
            raise e
        Logger.log('✨ all resources unpacked')