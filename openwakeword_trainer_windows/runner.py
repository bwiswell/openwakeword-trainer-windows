import os
import sys
import subprocess
from typing import Optional

from .config import Config
from .data_manager import DataManager
from .logger import Logger
from .tts import TTS
from .util import patch_all


class Runner:

    def __init__ (
                self,
                model: str,
                data_dir: Optional[str] = DataManager.DEFAULT_DATA_PATH,
                output_dir: Optional[str] = DataManager.DEFAULT_OUTPUT_PATH
            ):
        self.dm = DataManager(model, data_dir, output_dir)
        self.config: Config = None
        

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

    def _configure (self):
        Logger.start_phase('Creating Config')
        Logger.log('🚀 creating configuration file...')
        self.config = Config(self.dm)
        Logger.log('✨ configuration file created')

    def _download (self):
        Logger.start_phase('Downloading Resources')
        self.dm.download()

    def _ensure (self):
        Logger.start_phase('Ensuring Paths')
        self.dm.ensure_paths()

    def _patch (self):
        Logger.start_phase('Applying Patches')
        Logger.log('🚀 patching dependencies...')
        patch_all()
        Logger.log('✨ dependencies patched')

    def _train (self):
        Logger.start_phase('Training Model')
        Logger.log('🚀 model training...')
        subprocess.run([
            sys.executable, str(DataManager.SCRIPT_PATH),
            '--training_config', str(self.dm.train_conf_path),
            '--train_model'
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
    def run (self):
        self._patch()
        self._ensure()
        self._download()
        self._unpack()
        self._configure()
        self._tts()
        self._augment()
        self._train()