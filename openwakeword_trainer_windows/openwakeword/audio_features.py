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


from collections import deque
from multiprocessing.pool import ThreadPool
from typing import Deque, Optional

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from ..data_manager import DataManager


class AudioFeatures:
    '''
    A class for creating audio features from audio data, including
    melspectograms and Google's `speech_embedding` features.
    '''
    def __init__(self, n_cpus: int):
        self.n_cpus = n_cpus

        self.embedding_path = DataManager.MODEL_PATH / 'embedding_model.onnx'
        self.melspec_path = DataManager.MODEL_PATH / 'melspectrogram.onnx'

        options = ort.SessionOptions()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.embedding_model = ort.InferenceSession(
            self.embedding_path,
            options,
            providers
        )
        self.melspec_model = ort.InferenceSession(
            self.melspec_path,
            options,
            providers
        )

        # Create databuffers with empty/random data
        # Sample rate * 10
        self.raw_data_buffer: Deque = deque(maxlen=160000)
        # n_frames x n_features
        self.melspectrogram_buffer = np.ones((76, 32))
        # 10 seconds * 97 fps
        self.melspectrogram_max_len = 970
        self.accumulated_samples = 0
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer = self._audio_embeddings(
            np.random.randint(-1000, 1000, 16000*4).astype(np.int16)
        )
        # ~10 seconds of feature buffer history
        self.feature_buffer_max_len = 120


    ### PROPERTIES ###
    @property
    def execution_provider (self) -> str:
        return self.embedding_model.get_providers()[0]


    ### MAGIC METHODS ###
    def __call__(self, x):
        return self._streaming_features(x)
    

    ### HELPERS ###
    def _audio_embeddings(
                self,
                x: npt.NDArray[np.int16]
            ) -> npt.NDArray[np.float32]:
        '''
        Returns embeddings computed from the provided audio samples.
        '''
        spec = self._melspec(x)
        windows = []
        for i in range(0, spec.shape[0], 8):
            window = spec[i:i + 76]
            if window.shape[0] == 76:
                windows.append(window)

        batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
        embedding = self._embeddings_predict(batch)
        return embedding
    

    def _buffer_raw_data (self, x: npt.NDArray[np.int16]):
        '''
        Adds raw audio data to the input buffer.
        '''
        self.raw_data_buffer.extend(x.tolist())
    

    def _embeddings_batch(
                self,
                x: npt.NDArray[np.float32],
                batch_size: int = 128,
                n_cpus: int = 1
            ) -> npt.NDArray[np.float32]:
        """
        Returns embeddings for the input melspectrograms in batches.

        Parameters:
            x (`ndarray`):
                A numpy array of melspectrograms of shape
                (`n`, frames, melbins). Assumes that all of the melspectrograms
                have the same shape.
            batch_size (`int`):
                The batch size to use when computing the embeddings.
            n_cpus (`int`):
                The number of CPUs to use when computing the embeddings. This
                argument has no effect if the underlying model is executing on
                a GPU.

        Returns:
            embeddings (`ndarray`):
                A numpy array of shape (`n`, frames, embedding_dim) containing
                the embeddings of all `n` input melspectrograms.
        """
        # Prepare ThreadPool object, if needed for multithreading
        pool: Optional[ThreadPool] = None
        if "CPU" in self.execution_provider:
            pool = ThreadPool(processes = n_cpus)

        # Calculate array sizes and make batches
        n_frames = (x.shape[1] - 76)//8 + 1
        embedding_dim = 96
        embeddings = np.empty(
            (x.shape[0], n_frames, embedding_dim),
            dtype=np.float32
        )

        batch = []
        ndcs = []
        for ndx, melspec in enumerate(x):
            window_size = 76
            for i in range(0, melspec.shape[0], 8):
                window = melspec[i:i+window_size]
                if window.shape[0] == window_size:
                    batch.append(window)
            ndcs.append(ndx)

            if len(batch) >= batch_size or ndx+1 == x.shape[0]:
                batch = np.array(batch).astype(np.float32)
                if "CUDA" in self.execution_provider:
                    result = self._embeddings_predict(batch)

                elif pool:
                    if batch.shape[0] >= n_cpus:
                        chunksize = batch.shape[0] // n_cpus
                    else:
                        chunksize = 1
                    result = np.array(
                        pool.map(
                            self._melspec_embeddings,
                            batch,
                            chunksize=chunksize
                        )
                    )

                for j, ndx2 in zip(range(0, result.shape[0], n_frames), ndcs):
                    embeddings[ndx2, :, :] = result[j:j+n_frames]

                batch = []
                ndcs = []

        # Cleanup ThreadPool
        if pool:
            pool.close()

        return embeddings
        

    def _embeddings_predict (
                self,
                x: npt.NDArray[np.float32]
            ) -> npt.NDArray[np.float32]:
        return self.embedding_model.run(None, { 'input_1': x})[0].squeeze()


    def _melspec (self, x: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
        '''
        Function to compute the mel-spectrogram of the provided audio samples.

        Parameters:
            x (`ndarray`):
                The input audio data to compute the melspectrogram from.

        Return:
            melspectrogram (`ndarray`):
                The computed melspectrogram of the input audio data.
        '''
        x = x[None, ] if len(x.shape) < 2 else x
        outputs = self._melspec_predict(x.astype(np.float32))
        spec: npt.NDArray[np.float32] = np.squeeze(outputs[0]) / 10 + 2
        return spec


    def _melspec_batch(
                self,
                x: npt.NDArray[np.int16],
                batch_size: int = 128,
                n_cpus: int = 1
            ) -> npt.NDArray[np.float32]:
        """
        Compute the melspectrogram of the input audio samples in batches.

        Args:
            x (`ndarray`):
                A numpy array of 16 khz input audio data in shape
                (`n`, samples). Assumes that all of the audio data is the same
                length (same number of samples).
            batch_size (`int`):
                The batch size to use when computing the melspectrogram.
            n_cpus (`int`):
                The number of CPUs to use when computing the melspectrogram.
                This argument has no effect if the underlying model is
                executing on a GPU.

        Returns:
            melspecs (`ndarray`):
                A numpy array of shape (`n`, frames, melbins) containing the
                melspectrogram of all `n` input audio examples
        """

        # Prepare ThreadPool object, if needed for multithreading
        pool: Optional[ThreadPool] = None
        if "CPU" in self.execution_provider:
            pool = ThreadPool(processes = n_cpus)

        # Make batches
        n_frames = int(np.ceil(x.shape[1]/160 - 3))
        mel_bins = 32
        melspecs = np.empty((x.shape[0], n_frames, mel_bins), dtype=np.float32)
        for i in range(0, max(batch_size, x.shape[0]), batch_size):
            batch = x[i:i+batch_size]

            if "CUDA" in self.execution_provider:
                result = self._melspec(batch)

            elif pool:
                if batch.shape[0] >= n_cpus:
                    chunksize = batch.shape[0] // n_cpus
                else:
                    chunksize = 1
                result = np.array(
                    pool.map(
                        self._melspec,
                        batch,
                        chunksize=chunksize
                    )
                )

            melspecs[i:i+batch_size, :, :] = result.squeeze()

        # Cleanup ThreadPool
        if pool:
            pool.close()

        return melspecs


    def _melspec_embeddings (
                self,
                melspec: npt.NDArray[np.float32]
            ) -> npt.NDArray[np.float32]:
        '''
        Returns the Google `speech_embedding` features from a melspectrogram
        input.

        Parameters:
            melspec (`ndarray`):
                The input melspectrogram.

        Returns:
            **embeddings** (`ndarray`):
                The computed audio features/embeddings.
        '''
        if melspec.shape[0] != 1:
            melspec = melspec[None, ]
        embedding = self._embeddings_predict(melspec)
        return embedding
    

    def _melspec_predict (
                self,
                x: npt.NDArray[np.float32]
            ) -> npt.NDArray[np.float32]:
        return self.melspec_model.run(None, { 'input': x })


    def _reset(self):
        self.raw_data_buffer.clear()
        self.melspectrogram_buffer = np.ones((76, 32))
        self.accumulated_samples = 0
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer = self._audio_embeddings(
            np.random.randint(-1000, 1000, 16000*4).astype(np.int16)
        )


    def _streaming_features (self, x: npt.NDArray[np.int16]) -> int:
        '''
        Adds raw audio data to the buffer, temporarily storing extra frames if
        there are not an even number of 80ms chunks.
        '''
        processed_samples = 0

        if self.raw_data_remainder.shape[0] != 0:
            x = np.concatenate((self.raw_data_remainder, x))
            self.raw_data_remainder = np.empty(0)

        if self.accumulated_samples + x.shape[0] >= 1280:
            remainder = (self.accumulated_samples + x.shape[0]) % 1280
            if remainder != 0:
                x_even_chunks = x[0:-remainder]
                self._buffer_raw_data(x_even_chunks)
                self.accumulated_samples += len(x_even_chunks)
                self.raw_data_remainder = x[-remainder:]
            elif remainder == 0:
                self._buffer_raw_data(x)
                self.accumulated_samples += x.shape[0]
                self.raw_data_remainder = np.empty(0)
        else:
            self.accumulated_samples += x.shape[0]
            self._buffer_raw_data(x)

        # Only calculate melspectrogram once minimum samples are accumulated
        if self.accumulated_samples >= 1280 and \
                self.accumulated_samples % 1280 == 0:
            self._streaming_melspectrogram(self.accumulated_samples)

            # Calculate new audio embeddings/features based on update
            # melspectrograms
            for i in np.arange(self.accumulated_samples//1280-1, -1, -1):
                ndx = -8*i
                ndx = ndx if ndx != 0 else len(self.melspectrogram_buffer)
                x = self.melspectrogram_buffer[-76 + ndx:ndx]
                x = x.astype(np.float32)[None, :, :, None]
                if x.shape[1] == 76:
                    self.feature_buffer = np.vstack(
                        (self.feature_buffer, self._embeddings_predict(x))
                    )

            # Reset raw data buffer counter
            processed_samples = self.accumulated_samples
            self.accumulated_samples = 0

        if self.feature_buffer.shape[0] > self.feature_buffer_max_len:
            start = -self.feature_buffer_max_len
            self.feature_buffer = self.feature_buffer[start:, :]

        if processed_samples != 0:
            return processed_samples
        else:
            return self.accumulated_samples


    def _streaming_melspectrogram (self, n_samples: int):
        self.melspectrogram_buffer = np.vstack((
            self.melspectrogram_buffer,
            self._melspec(
                np.array(
                    list(self.raw_data_buffer)[-n_samples-160*3:],
                    dtype=np.int16
                )
            )
        ))
        if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
            start = -self.melspectrogram_max_len
            self.melspectrogram_buffer = self.melspectrogram_buffer[start:, :]

    
    ### METHODS ###
    def embed_clips(
                self,
                x: npt.NDArray[np.int16],
                batch_size: int,
                n_cpus: int
            ) -> npt.NDArray[np.float32]:
        '''
        Compute the embeddings of the input audio clips in batches.

        Note that the optimal performance will depend in the interaction
        between the device, batch size, and ncpu (if a CPU device is used). The
        user is encouraged to experiment with different values of these
        parameters to identify which combination is best for their data, as
        often differences of 1-4x are seen.

        Parameters:
            x (`ndarray`):
                A numpy array of 16 khz input audio data in shape
                (`n`, samples). Assumes that all of the audio data is the same
                length (same number of samples).
            batch_size (`int`):
                The batch size to use when computing the embeddings
            n_cpus (`int`):
                The number of CPUs to use when computing the melspectrogram.
                This argument has no effect if the underlying model is
                executing on a GPU.

        Returns:
            embeddings (`ndarray`):
                A numpy array of shape (`n`, frames, embedding_dim) containing
                the embeddings of all `n` input audio clips
        '''
        melspecs = self._melspec_batch(
            x,
            batch_size = batch_size,
            n_cpus = n_cpus
        )
        embeddings = self._embeddings_batch(
            melspecs[:, :, :, None],
            batch_size = batch_size,
            n_cpus = n_cpus
        )

        return embeddings
    
    
    def get_embedding_shape (self, audio_length: float) -> tuple[int]:
        '''
        Returns the size of the output embedding array for a given audio clip
        length (in seconds).
        '''
        n_samples = int(audio_length * 16000)
        x = (np.random.uniform(-1, 1, n_samples) * 32767).astype(np.int16)
        return self._audio_embeddings(x).shape