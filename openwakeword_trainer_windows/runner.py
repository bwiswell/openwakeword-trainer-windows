import sys
import subprocess
from typing import Optional

from .config import Config
from .data_manager import DataManager
from .logger import Logger
from .pipeline_step import PipelineStep
from .recorder import Recorder
from .tts import TTS


class Runner:

    def __init__ (
                self,
                model: str,
                data_dir: Optional[str] = DataManager.DEFAULT_DATA_PATH,
                output_dir: Optional[str] = DataManager.DEFAULT_OUTPUT_PATH
            ):
        self.dm = DataManager(model, data_dir, output_dir)
        self.config = Config(self.dm.config_path)
        

    ### HELPERS ###
    def _augment (self):
        Logger.start_phase('Augmenting Data')
        Logger.log('🚀 starting data augmentation...')
        subprocess.run([
            sys.executable, str(DataManager.SCRIPT_PATH),
            '--training_config', str(self.dm.train_conf_path),
            '--augment_clips'
        ], check=True)
        Logger.log('✨ data augmentation complete')

    def _download (self):
        Logger.start_phase('Downloading Resources')
        self.dm.download()

    def _ensure (self):
        Logger.start_phase('Ensuring Paths')
        self.dm.ensure_paths()

    def _export (self):
        Logger.start_phase('Exporting Models')
        self.dm.export()

    def _record (self):
        Logger.start_phase('Recording Samples')
        recorder = Recorder(self.config, self.dm)
        recorder.record_samples()

    def _train (self):
        Logger.start_phase('Training Model')
        Logger.log('🚀 model training...')
        subprocess.run([
            sys.executable, str(DataManager.SCRIPT_PATH),
            '--training_config', str(self.dm.train_conf_path),
            '--train_model'#, '--convert_to_tflite'
        ], check=True)
        Logger.log('✨ training complete')

    def _tts (self):
        Logger.start_phase('Generating TTS Samples')
        t = TTS()
        t.generate(self.config, self.dm)

    def _unpack (self):
        Logger.start_phase('Unpacking Resources')
        self.dm.unpack()


    ### METHODS ###
    def run (
                self,
                start_from: PipelineStep = PipelineStep.ENSURE,
                end_at: PipelineStep = PipelineStep.EXPORT,
                do_only: Optional[PipelineStep] = None
            ):
        start = start_from.value if do_only is None else do_only.value
        end = end_at.value if do_only is None else do_only.value

        for i in range(start, end + 1):
            match PipelineStep(i):
                case PipelineStep.ENSURE:
                    self._ensure()
                case PipelineStep.DOWNLOAD:
                    self._download()
                case PipelineStep.UNPACK:
                    self._unpack()
                case PipelineStep.RECORD:
                    self._record()
                case PipelineStep.TTS:
                    self._tts()
                case PipelineStep.AUGMENT:
                    self._augment()
                case PipelineStep.TRAIN:
                    self._train()
                case PipelineStep.EXPORT:
                    self._export()