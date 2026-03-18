import itertools as it
from pathlib import Path

import kokoro as ko
import numpy as np
import soundfile as sf
import torchaudio as ta
import tqdm as tq

from .config import Config
from .data_manager import DataManager
from .logger import Logger


class TTS:

    VOICES = [
        'af_heart',
        'af_alloy',
        'af_aoede',
        'af_bella',
        'af_jessica',
        'af_kore',
        'af_nicole',
        'af_nova',
        'af_river',
        'af_sarah',
        'af_sky',
        'am_adam',
        'am_echo',
        'am_eric',
        'am_fenrir',
        'am_liam',
        'am_michael',
        'am_onyx',
        'am_puck',
        'am_santa'
    ]


    def __init__ (self):
        self.pipeline = ko.KPipeline(
            lang_code = 'a',
            repo_id = 'hexgrad/Kokoro-82M'
        )
        self.resamplers = [
            ta.transforms.Resample(24000, 15000 + i * 500)
            for i in range(5)
        ]


    ### HELPERS ###
    def _generate (
                self,
                index: int,
                output: Path,
                phrase: str,
                voice: str,
                speed: float,
                pbar: tq.tqdm
            ):
        generator = self.pipeline(phrase, voice, speed)
        _, _, audio = next(generator)
        for i, resampler in enumerate(self.resamplers):
            resampled = resampler(audio)
            path = str(output / f'sample_{index + i}.wav')
            sf.write(path, resampled, 16000)
            pbar.update()
    
    def _generate_batch (
                self,
                index: int,
                output: Path,
                phrase: str,
                speeds: list[float]
            ):
        step = len(speeds)
        n = 100 * step
        pbar = tq.tqdm(total=n, desc=f'Generating samples for "{phrase}"')
        for i, (voice, speed) in enumerate(it.product(TTS.VOICES, speeds)):
            self._generate(
                index + i * step,
                output,
                phrase,
                voice,
                speed,
                pbar
            )

    def _generate_split (
                self,
                output: Path,
                phrases: list[str],
                samples_per_phrase: int
            ):
        n_speeds = samples_per_phrase // 100
        speeds = list(np.linspace(0.7, 1.3, n_speeds))
        for i, phrase in enumerate(phrases):
            self._generate_batch(
                i * samples_per_phrase,
                output,
                phrase,
                speeds
            )


    ### METHODS ###
    def generate (self, config: Config, dm: DataManager):
        Logger.log('🚀 starting sample generation...')

        splits = [
            'positive training',
            'positive testing',
            'negative training',
            'negative testing'
        ]

        phase_phrases = [
            config.target_phrases,
            config.target_phrases,
            config.negative_phrases,
            config.negative_phrases
        ]

        n_train_per_pos = config.n_train // len(config.target_phrases)
        n_test_per_pos = config.n_test // len(config.target_phrases)
        n_train_per_neg = config.n_train // len(config.negative_phrases)
        n_test_per_neg = config.n_test // len(config.negative_phrases)
        counts = [
            n_train_per_pos,
            n_test_per_pos,
            n_train_per_neg,
            n_test_per_neg
        ]

        paths = [
            dm.pos_train,
            dm.pos_test,
            dm.neg_train,
            dm.neg_test
        ]

        zipped = zip(splits, phase_phrases, counts, paths)
        for s_type, phrases, count, path in zipped:
            Logger.log(f'🔄 generating {s_type} samples...')
            self._generate_split(
                path,
                phrases,
                count
            )

        Logger.log('✨ all samples generated')