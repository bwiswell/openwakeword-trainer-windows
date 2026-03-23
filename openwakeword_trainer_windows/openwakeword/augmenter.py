# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################

# NOTICE: This file has been substantially modified by Benj Wiswell.


from dataclasses import dataclass
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Generator
from uuid import uuid4

import audiomentations as am
import numpy as np
from numpy.lib.format import open_memmap
import numpy.typing as npt
import scipy.io.wavfile as siw
from speechbrain.processing.signal_processing import reverberate
import torch as to
import torch_audiomentations as tam
import torchaudio as ta
import tqdm as tq

from ..config import Config
from ..data_manager import DataManager
from ..logger import Logger

from .audio_features import AudioFeatures


@dataclass
class AugmentationProbabilities:

    background_noise: float
    band_stop_filter: float
    colored_noise: float
    gain: float
    pitch_shift: float
    rir: float
    seven_band_parametric_eq: float
    tanh_distortion: float


class Augmenter:

    AUG_PROBS = AugmentationProbabilities(
        background_noise = 0.75,
        band_stop_filter = 0.25,
        colored_noise = 0.25,
        gain = 1.0,
        pitch_shift = 0.25,
        rir = 0.5,
        seven_band_parametric_eq = 0.25,
        tanh_distortion = 0.25
    )


    def __init__ (self, config: Config, dm: DataManager):
        self.config = config
        self.dm = dm


    ### HELPERS ###
    def _augment_batch (
                self,
                name: str,
                path: str,
                total_length: int,
                background_clip_paths: list[str],
                rir_paths: list[str],
                n_cpus: int
            ):
        Logger.log(f'🔄 augmenting {name} features...')
        clips, count = self._get_clips(path)
        generator = self._augment_clips(
            clips,
            total_length,
            self.config.augmentation_batch,
            background_clip_paths,
            rir_paths
        )
        temp_path = self._get_temp()
        self._compute_features(
            name,
            generator,
            count,
            total_length,
            temp_path,
            n_cpus
        )
        self._trim(temp_path)
        Logger.log(f'✅ {name} successfully augmented')
        Logger.log(f'🔄 writing {name} features...')
        if os.path.exists(path): os.remove(path)
        shutil.move(temp_path, path)
        Logger.log(f'✅ {name} successfully written')


    def _augment_clips (
                self,
                clip_paths: list[str],
                total_length: int,
                batch_size: int,
                background_clip_paths: list[str],
                rir_paths: list[str]
            ) -> Generator[npt.NDArray[np.int16], None, None]:
        '''
        Applies audio augmentations to the specified audio clips, returning a
        generator that applies the augmentations in batches to support very
        large quantities of input audio files.

        The augmentations (and probabilities) are chosen from experience based
        on training openWakeWord models, as well as for the efficiency of the
        augmentation.

        Args:
            clip_paths (`list[str]`): 
                The input audio files (as paths) to augment. Note that these
                should be shorter than the "total_length" argument, else they
                will be truncated.
            total_length (`int`):
                The total length of audio files (in samples) after
                augmentation. All input clips will be left-padded with silence
                to reach this size, with between 0 and 200 ms of other audio
                after the end of the original input clip.
            batch_size (`int`):
                The number of audio files to augment at once.
            background_clip_paths (`list[str]`):
                The paths to background audio files to mix with the input
                files.
            rir_paths (`list[str]`):
                The paths to room impulse response functions (RIRs) to convolve
                with the input files, producing a version of the input clip
                with different acoustic characteristics.

        Returns:
            augmented (`ndarray`): 
                A batch of augmented audio clips of size
                (batch_size, total_length).
        '''

        # First pass augmentations that can't be done as a batch
        aug_a = am.Compose([
            am.SevenBandParametricEQ(
                min_gain_db = -6,
                max_gain_db = 6,
                p = Augmenter.AUG_PROBS.seven_band_parametric_eq
            ),
            am.TanhDistortion(
                min_distortion = 0.0001,
                max_distortion = 0.10,
                p = Augmenter.AUG_PROBS.tanh_distortion
            ),
        ])

        # Augmentations that can be done as a batch
        aug_b = tam.Compose([
            tam.PitchShift(
                min_transpose_semitones = -3,
                max_transpose_semitones = 3,
                p = Augmenter.AUG_PROBS.pitch_shift,
                sample_rate = 16000,
                mode = "per_batch"
            ),
            tam.BandStopFilter(
                p = Augmenter.AUG_PROBS.band_stop_filter,
                mode = "per_batch"
            ),
            tam.AddColoredNoise(
                min_snr_in_db = 10,
                max_snr_in_db = 30,
                min_f_decay = -1,
                max_f_decay = 2,
                p = Augmenter.AUG_PROBS.colored_noise,
                mode = "per_batch"
            ),
            tam.AddBackgroundNoise(
                p = Augmenter.AUG_PROBS.background_noise,
                background_paths = background_clip_paths,
                min_snr_in_db = -10,
                max_snr_in_db = 15,
                mode = "per_batch"
            ),
            tam.Gain(
                max_gain_in_db = 0,
                p = Augmenter.AUG_PROBS.gain
            ),
        ])

        # Iterate through all clips and augment them
        for i in range(0, len(clip_paths), batch_size):
            batch = clip_paths[i:i+batch_size]
            augmented_clips = []
            for clip in batch:
                clip_data, clip_sr = ta.load(clip)
                clip_data = clip_data[0]
                if clip_data.shape[0] > total_length:
                    clip_data = clip_data[0:total_length]

                clip_data = self._fixed_size_clip(
                    clip_data,
                    total_length,
                    clip_sr
                )

                # Do first pass augmentations
                augmented_clips.append(
                    to.from_numpy(
                        aug_a(
                            samples = clip_data,
                            sample_rate = 16000
                        )
                    )
                )

            # Do second pass augmentations
            device = to.device('cuda:0' if to.cuda.is_available() else 'cpu')
            augmented_batch = aug_b(
                samples = to.vstack(
                    augmented_clips
                ).unsqueeze(dim=1).to(device),
                sample_rate = 16000
            ).squeeze(axis=1)

            # Do reverberation
            if Augmenter.AUG_PROBS.rir >= np.random.random():
                rir_waveform, _ = ta.load(random.choice(rir_paths))
                augmented_batch = reverberate(
                    augmented_batch.cpu(),
                    rir_waveform,
                    rescale_amp = 'avg'
                )

            # yield batch of 16-bit PCM audio data
            yield (augmented_batch.cpu().numpy() * 32767).astype(np.int16)


    def _compute_features (
                self,
                name: str,
                generator: Generator[npt.NDArray[np.int16], None, None],
                n_total: int,
                total_length: int,
                output_path: str,
                n_cpus: int
            ):
        if to.cuda.is_available():
            device = 'gpu'
            n_cpus = 1
        else:
            device = 'cpu'

        af = AudioFeatures(device=device, model_path=self.dm.resource_path)
        n_cols = af.get_embedding_shape(total_length / 16000)
        out_shape = (n_total, *n_cols)
        fp = open_memmap(
            output_path,
            mode='w+',
            dtype=np.float32,
            shape=out_shape
        )

        row_idx = 0
        audio_data = next(generator)
        batch_size = audio_data.shape[0]

        features = af.embed_clips(audio_data, batch_size, 1)
        fp[row_idx:row_idx + features.shape[0], :, :] = features
        row_idx += features.shape[0]
        fp.flush()

        pbar = tq.tqdm(
            generator,
            total = n_total // batch_size,
            desc = f'computing {name} features'
        )
        for audio_data in pbar:
            if row_idx >= n_total:
                break

            features = af.embed_clips(audio_data, batch_size, n_cpus)
            if row_idx + features.shape[0] > n_total:
                features = features[0:n_total - row_idx]

            fp[row_idx:row_idx + features.shape[0], :, :] = features
            row_idx += features.shape[0]
            fp.flush()


    def _find_total_length (self):
        n = 50  # sample size
        positive_clips = [str(i) for i in Path(self.dm.pos_test).glob("*.wav")]
        duration_in_samples = []
        for _ in range(n):
            _, dat = siw.read(
                positive_clips[np.random.randint(0, len(positive_clips))]
            )
            duration_in_samples.append(len(dat))

        length = int(round(np.median(duration_in_samples)/1000)*1000) + 12000
        if length < 32000 or abs(length - 32000) <= 4000:
            length = 32000

        return length


    def _fixed_size_clip(
                x: to.FloatTensor,
                n_samples: int
            ) -> to.FloatTensor:
        '''
        Create a fixed-length clip of the specified size by padding an input
        clip with zeros.

        Parameters:
            x (`FloatTensor`):
                The input audio to pad to a fixed size.
            n_samples (`int`):
                The total number of samples for the fixed length clip.

        Returns:
            **clipped** (`ndarray`):
                A new array of audio data of the specified length
        '''
        if x.dim() > 1:
            x = x.squeeze()

        if len(x) > n_samples:
            if np.random.random() >= 0.5:
                return x[:x.shape[0]]
            else:
                return x[-x.shape[0]:]

        max_jitter = int(np.random.uniform(0, 0.2) * 16000)
        room = n_samples - x.shape[0]
        jitter = np.random.randint(0, max_jitter + 1)
        start = max(0, room - jitter)

        dat = to.zeros(n_samples, dtype=x.dtype, device=x.device)
        dat[start:start + x.shape[0]] = x
        return dat
    

    def _get_background_paths (self) -> list[str]:
        paths: list[str] = []
        for path in self.config.background_paths:
            paths.extend([i.path for i in os.scandir(path)])
        return paths


    def _get_clips (self, path: Path) -> tuple[list[str], int]:
        paths = [str(i) for i in path.glob('*.wav')]
        count = len(paths)
        return paths * self.config.augmentations, count
    

    def _get_rir_paths (self) -> list[str]:
        return [i.path for i in os.scandir(self.config.rir_path)]
    

    def _get_temp (self, name: str):
        return os.path.join(tempfile.gettempdir(), f'{uuid4()}.npy')
    

    def _trim (self, path: str):
        data = np.load(path)
        
        non_zero_rows = np.where(~np.all(data == 0, axis=(1, 2)))[0]
        if len(non_zero_rows) == 0:
            return
        
        last_row_index = non_zero_rows[-1] + 1
        trimmed_data = data[:last_row_index]
        
        del data
        time.sleep(2.0)
        np.save(path, trimmed_data)


    ### METHODS ###
    def augment (self):
        total_length = self._find_total_length()
        bg_paths = self._get_background_paths()
        rir_paths = self._get_rir_paths()
        
        n_cpus = os.cpu_count()
        if n_cpus is None:
            n_cpus = 1
        else:
            n_cpus = n_cpus // 2
        
        self._augment_batch(
            'positive training',
            self.dm.features.pos_train,
            total_length,
            bg_paths,
            rir_paths,
            n_cpus
        )

        self._augment_batch(
            'positive testing',
            self.dm.features.pos_test,
            total_length,
            bg_paths,
            rir_paths,
            n_cpus
        )

        self._augment_batch(
            'negative training',
            self.dm.features.neg_train,
            total_length,
            bg_paths,
            rir_paths,
            n_cpus
        )

        self._augment_batch(
            'negative testing',
            self.dm.features.neg_test,
            total_length,
            bg_paths,
            rir_paths,
            n_cpus
        )