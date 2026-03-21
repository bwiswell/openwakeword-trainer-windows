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

#######################
# Silero VAD License
#######################

# MIT License

# Copyright (c) 2020-present Silero Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

########################################

# This file contains the implementation of a class for voice activity detection (VAD),
# based on the pre-trained model from Silero (https://github.com/snakers4/silero-vad).
# It can be used as with the openWakeWord library, or independently.
#
# NOTICE: This file has been substantially modified by Benj Wiswell.


from collections import deque

import onnxruntime as ort
import numpy as np
import numpy.typing as npt

from ..data_manager import DataManager


class VAD():
    '''
    A model class for a voice activity detection (VAD) based on Silero's model:

    https://github.com/snakers4/silero-vad
    '''
    def __init__(self, dm: DataManager):
        # Initialize the ONNX model
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = 1
        sessionOptions.intra_op_num_threads = 1
        self.model = ort.InferenceSession(
            str(dm.MODEL_PATH / 'silero_vad.onnx'),
            sess_options = sessionOptions,
            providers = ["CPUExecutionProvider"]
        )

        # Create buffer
        self.prediction_buffer: deque = deque(maxlen=125)

        # Set model parameters
        self.sample_rate = np.array(16000).astype(np.int64)

        # Reset model to start
        self._reset_states()


    ### MAGIC METHODS ###
    def __call__(self, x: npt.NDArray[np.float_], frame_size: int = 480):
        self.prediction_buffer.append(self._predict(x, frame_size))


    ### HELPERS ###
    def _reset_states(self, batch_size: int = 1):
        self._h = np.zeros((2, batch_size, 64)).astype('float32')
        self._c = np.zeros((2, batch_size, 64)).astype('float32')
        self._last_sr = 0
        self._last_batch_size = 0


    def _predict(self, x: npt.NDArray[np.float_], frame_size: int = 480):
        '''
        Get the VAD predictions for the input audio frame.

        Parameters:
            x (`ndarray`):
                The input audio, must be 16 khz and 16-bit PCM format. If
                longer than the input frame, the audio will be split into
                chunks of length `frame_size` and the average prediction across
                all chunks will be returned. Must be a length that is an
                integer multiple of the `frame_size` argument.
            frame_size (`int`):
                The frame size in samples. The recommended default is 480
                samples (30 ms @ 16khz), but smaller and larger values can be
                used (though performance may decrease).

        Returns
            **score** (`float`):
                The average predicted score for the audio frame.
        '''
        chunks = [(x[i:i+frame_size]/32767).astype(np.float32)
                  for i in range(0, x.shape[0], frame_size)]

        frame_predictions = []
        for chunk in chunks:
            ort_inputs = {'input': chunk[None, ],
                          'h': self._h, 'c': self._c, 'sr': self.sample_rate}
            ort_outs = self.model.run(None, ort_inputs)
            out, self._h, self._c = ort_outs
            frame_predictions.append(out[0][0])

        return np.mean(frame_predictions)
