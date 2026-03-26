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


from collections import deque, defaultdict
import functools as ft
import os
from pathlib import Path
import pickle
import time
from typing import Any, Callable, DefaultDict, Literal, Optional, Union
import wave

import numpy as np
import numpy.typing as npt
import sklearn.linear_model as skl

from .audio_features import AudioFeatures
from .common import (
    InferenceFramework,
    IncompatibleModelError,
    InferenceFrameworkError,
    ModelNameConflictError,
    ModelNotFoundError,
    UnknownVerifierModelError,
    WrongPredictMethodError
)
from .downloader import Downloader
from .vad import VAD


RuntimeModel = Any
ModelDict = dict[str, RuntimeModel]
PredictFn = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
PredictDict = dict[str, PredictFn]

PredictResults = Union[
    dict[str, float],
    tuple[dict[str, float], dict[str, float]]
]


class Model():
    '''
    The main model class for openWakeWord. Creates a model object with the
    shared audio pre-processer and for arbitrarily many custom wake word/wake
    phrase models.
    '''

    ALL = [
        'alexa',
        'hey_jarvis',
        'hey_mycroft',
        'hey_rhasspy', 
        'timer',
        'weather'
    ]

    DEFAULT_MODEL_PATH = Path(__file__).parent.absolute() / 'resources'

    TIMER_MAPPINGS = {
        '1': '1_minute_timer',
        '2': '5_minute_timer',
        '3': '10_minute_timer',
        '4': '20_minute_timer',
        '5': '30_minute_timer',
        '6': '1_hour_timer'
    }

    def __init__ (
                self,
                custom_models: dict[str, str] = {},
                pretrained_models: list[str] = [],
                vad_threshold: float = 0,
                custom_verifier_models: dict[str, str] = {},
                custom_verifier_threshold: float = 0.1,
                inference_framework: InferenceFramework = "tflite",
                model_path: Path = DEFAULT_MODEL_PATH
            ):
        '''
        Initialize the openWakeWord model object.

        Parameters:
            custom_models (`dict[str, str]`):
                A `dict` mapping custom model names to custom model paths. If
                not provided, model loading falls back to the
                `pretrained_models` parameter.
            pretrained_models (`list[str]`):
                A `list` of pretrained model names ('alexa', 'hey_jarvis',
                'hey_mycroft', 'hey_rhasspy', 'timer', or 'weather') to load.
                The models are automatically downloaded if not already present
                at `model_path`. If `custom_models` and `pretrained_models` are
                both empty, then all pretrained models will be loaded. If
                values are passed to `custom_models`, no pretrained models are
                loaded unless explicitly specified. The `Model.ALL` class
                property is provided for loading all pretrained models
                explicitly, i.e. `Model(custom_models = 
                { 'some_name': 'some_path' }, pretrained_models = Model.ALL)`.
            vad_threshold (`float`):
                Whether to use a voice activity detection model (VAD) from
                Silero (https://github.com/snakers4/silero-vad) to filter
                predictions. For every input audio frame, a VAD score is
                obtained and only those model predictions with VAD scores above
                the threshold will be returned. The default value (0), disables
                voice activity detection entirely. VAD is only available with
                the `onnx` inference framework.
            custom_verifier_models (`dict[str, str]`):
                A `dict` of paths to custom verifier models, where the keys are
                the model names (corresponding to the provided 
                custom/pretrained model names) and the values are the filepaths
                of the custom verifier models.
            custom_verifier_threshold (`float`):
                The score threshold to use a custom verifier model. If the
                score from a model for a given frame is greater than this
                value, the associated custom verifier model will also predict
                on that frame, and the verifier score will be returned.
            inference_framework (`InferenceFramework`):
                The inference framework to use when for model prediction.
                Options are `tflite` or `onnx`. The default is `tflite` as this
                results in better efficiency on common platforms (x86, ARM64),
                but in some deployment scenarios ONNX models may be preferable.
            model_path (`Path`):
                The path where pretrained openWakeWord models and the 
                `embedding_model` and `melspectrogram` models for audio
                preprocessing are located or should be automatically downloaded
                to. Defaults to a local 'resources' directory that will be
                automatically created if it does not yet exist.
        '''

        # Ensure options are valid
        if inference_framework not in ('onnx', 'tflite'):
            raise InferenceFrameworkError(inference_framework)

        if vad_threshold > 0 and inference_framework != 'onnx':
            raise IncompatibleModelError('VAD', inference_framework)
        
        for name, path in custom_models.items():
            framework = os.path.splitext(os.path.basename(path))[1]
            if framework != inference_framework:
                raise IncompatibleModelError(name, inference_framework)
            elif not os.path.exists(path):
                raise ModelNotFoundError(path, True, False)
            elif name in Model.ALL:
                raise ModelNameConflictError(name)
            
        if len(custom_models) + len(pretrained_models) == 0:
            pretrained_models = Model.ALL

        for name, path in custom_verifier_models.items():
            if name not in custom_models and name not in pretrained_models:
                raise UnknownVerifierModelError(name)
            elif not os.path.exists(path):
                raise ModelNotFoundError(path, True, True)
            
        self.model_paths = custom_models
            
        # Ensure pretrained models are downloaded
        try:
            downloader = Downloader(model_path, inference_framework)
            downloader.ensure_preprocessors()
            if vad_threshold > 0:
                downloader.ensure_vad()
            paths = downloader.ensure_models(pretrained_models)
            for name, path in zip(pretrained_models, paths):
                self.model_paths[name] = str(path)
        except Exception as e:
            raise e

        # Create attributes to store models and metadata
        if len(self.model_paths) == 1:
            self.is_singleton = True
            self.singleton = list(self.model_paths.keys())[0]
        else:
            self.is_singleton = False
        self.models: ModelDict = {}
        self.predict_fns: PredictDict = {}
        self.model_inputs: dict[str, int] = {}

        # Setup models and predict functions
        if inference_framework == 'onnx':
            self._setup_onnx()
        elif inference_framework == 'tflite':
            self._setup_tflite()

        # Initialize Silero VAD
        self.vad_threshold = vad_threshold
        if vad_threshold > 0:
            self.vad = VAD(model_path)

        # Load custom verifier models
        self.custom_verifiers: dict[str, skl.LogisticRegression] = {
            name: pickle.load(open(path, 'rb'))
            for name, path in custom_verifier_models.items()
        }
        self.custom_verifier_threshold = custom_verifier_threshold

        # Create buffer to store frame predictions
        self.prediction_buffer: DefaultDict[str, deque] = defaultdict(
            ft.partial(deque, maxlen=30)
        )

        # Create AudioFeatures object
        self.preprocessor = AudioFeatures(
            model_path = model_path,
            n_cpus = 1,
            inference_framework = inference_framework,
            device = 'cpu'
        )


    ### HELPERS ###
    def _predict (
                self,
                x: npt.NDArray[np.int16],
                model_patiences: dict[str, int] = {},
                model_thresholds: dict[str, float] = {},
                debounce_time: float = 0.0,
                timing: bool = False
            ) -> PredictResults:
        '''
        Predict with all of the wakeword models on the input audio frames and
        optionally gather timing information.

        Args:
            x (`ndarray`):
                The input audio data to predict on with the models. Ideally
                should be multiples of 80 ms (1280 samples), with longer
                lengths reducing overall CPU usage but decreasing detection
                latency. Input audio with durations greater than or less than
                80 ms is also supported, though this will add a detection delay
                of up to 80 ms as the appropriate number of samples are
                accumulated.
            model_patiences (`dict[str, int]`):
                How many consecutive frames (of 1280 samples or 80 ms) above
                the threshold that must be observed before the current frame
                will be returned as non-zero. Must be provided as an a `dict`
                where the keys are the model names and the values are the
                number of frames. Can reduce false-positive detections at the
                cost of a lower true-positive rate. By default, this behavior
                is disabled.
            model_thresholds (`dict[str, float]`):
                The threshold values to use when the `patience` or
                `debounce_time` behavior is enabled. Must be provided as a
                `dict` where the keys are the model names and the values are
                the thresholds.
            debounce_time (`float`):
                The time (in seconds) to wait before returning another non-zero
                prediction after a non-zero prediction. Can preven multiple
                detections of the same wake-word.
            timing (`bool`):
                Whether to return timing information of the models. Can be
                useful to debug and assess how efficiently models are running
                on the current hardware.

        Returns:
            results (`PredictResults`):
                Either a single `dict` mapping model names to prediction scores
                between 0 and 1, or a `tuple` of two `dicts`; the first mapping
                model names to prediction scores and the second mapping model
                names, 'preprocessing', and 'vad' keys to timing information.
        '''
        # Setup timing dict
        if timing:
            timings: dict[str, float] = {}
            feature_start = time.time()

        # Preprocess data
        x_preprocessed = self.preprocessor(x)
        if timing:
            timings['preprocessor'] = time.time() - feature_start

        # Get predictions from model(s)
        predictions = {}
        for model in self.models.keys():
            if timing:
                model_start = time.time()

            # Run model to get predictions
            if x_preprocessed > 1280:
                group_predictions = []
                for i in np.arange(x_preprocessed // 1280 - 1, -1, -1):
                    group_predictions.extend(
                        self.predict_fns(model)(
                            self.preprocessor.get_features(
                                    self.model_inputs[model],
                                    start_idx = -self.model_inputs[model] - i
                            )
                        )
                    )
                prediction = np.array(group_predictions).max(axis=0)[None, ]
            elif x_preprocessed == 1280:
                prediction = self.predict_fns[model](
                    self.preprocessor.get_features(self.model_inputs[model])
                )
            elif x_preprocessed < 1280:
                # get previous prediction if there aren't enough samples
                if model != 'timer':
                    if len(self.prediction_buffer[model]) > 0:
                        prediction = [[[self.prediction_buffer[model][-1]]]]
                    else:
                        prediction = [[[0]]]
                else:
                    n_classes = 6
                    prediction = [[[0] * (n_classes + 1)]]

            if model != 'timer':
                predictions[model] = prediction[0][0][0]
            else:
                for int_label, cls in Model.TIMER_MAPPINGS:
                    predictions[cls] = prediction[0][0][int(int_label)]

            # Update scores based on custom verifier model
            if len(self.custom_verifiers) > 0:
                for cls in predictions.keys():
                    if predictions[cls] >= self.custom_verifier_threshold:
                        if cls in Model.TIMER_MAPPINGS.values():
                            parent_model = 'timer'
                        else:
                            parent_model = model
                        if self.custom_verifiers.get(parent_model, False):
                            verifier = self.custom_verifiers[parent_model]
                            verifier_prediction = verifier.predict_proba(
                                self.preprocessor.get_features(
                                    self.model_inputs[model]
                                )
                            )[0][-1]
                            predictions[cls] = verifier_prediction

            # Zero predictions for first 5 frames during model initialization
            for cls in predictions.keys():
                if len(self.prediction_buffer[cls]) < 5:
                    predictions[cls] = 0.0

            # Get timing information
            if timing:
                timings[model] = time.time() - model_start

        # Update scores based on thresholds or patience arguments
        if len(model_patiences) > 0 or debounce_time > 0:
            if len(model_thresholds) < len(self.models):
                msg = "`model_patiences` or `debounce_time` was provided, \
                        but either no `model_thresholds` were provided, or \
                        not all models were given a threshold value."
                raise ValueError(msg)
            elif len(model_patiences) > 0 and debounce_time > 0:
                msg = "`model_patiences` and `debounce_time` may not be used \
                        together."
                raise ValueError(msg)
            elif len(model_patiences) > 0 and \
                    len(model_patiences) < len(self.models):
                msg = "`model_patiences` was provided, but not all models were \
                        given a patience value."
                raise ValueError(msg)
                
            for cls in predictions.keys():
                if cls in Model.TIMER_MAPPINGS.values():
                    parent_model = 'timer'
                else:
                    parent_model = model
                if predictions[cls] != 0.0:
                    threshold = model_thresholds[parent_model]
                    if parent_model in model_patiences.keys():
                        patience = model_patiences[parent_model]
                        scores = np.array(
                            self.prediction_buffer[cls]
                        )[-patience:]
                        if (scores >= threshold).sum() < patience:
                            predictions[cls] = 0.0
                    elif debounce_time > 0:
                        n_frames = int(
                            np.ceil(debounce_time / (x_preprocessed / 16000))
                        )
                        recent = np.array(
                            self.prediction_buffer[cls]
                        )[-n_frames:]
                        if predictions[cls] >= threshold and \
                                (recent >= threshold).sum() > 0:
                            predictions[cls] = 0.0

        # Update prediction buffer
        for cls in predictions.keys():
            self.prediction_buffer[cls].append(predictions[cls])

        # get voice activity detection scores and update model scores
        if self.vad_threshold > 0:
            if timing:
                vad_start = time.time()

            self.vad(x)

            if timing:
                timings['vad'] = time.time() - vad_start

            # Get frames from last 0.4 to 0.56 seconds (3 frames) before the
            # current frame and get max VAD score
            vad_frames = list(self.vad.prediction_buffer)[-7:-4]
            vad_max_score = np.max(vad_frames) if len(vad_frames) > 0 else 0
            for cls in predictions.keys():
                if vad_max_score < self.vad_threshold:
                    predictions[cls] = 0.0

        if timing:
            return predictions, timings
        else:
            return predictions
    

    def _setup_onnx (self):
        try:
            import onnxruntime as ort
        except ImportError:
            raise InferenceFrameworkError('onnx')

        def predict (
                    model: ort.InferenceSession,
                    x: npt.NDArray[np.float32]
                ) -> list[npt.NDArray[np.float32]]:
            model.run(None, { model.get_inputs()[0].name: x })

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        providers = ['CPUExecutionProvider']

        for name, path in self.model_paths.items():
            model = ort.InferenceSession(path, options, providers)
            self.models[name] = model
            self.predict_fns[name] = ft.partial(predict, model)
            self.model_inputs[name] = model.get_inputs()[0].shape[1]


    def _setup_tflite (self):
        try:
            import ai_edge_litert.interpreter as tflite
        except ImportError:
            raise InferenceFrameworkError('tflite')
        
        def predict (
                    model: tflite.Interpreter,
                    input_idx: int,
                    output_idx: int,
                    x: npt.NDArray[np.float32]
                ) -> npt.NDArray[np.float32]:
            model.set_tensor(input_idx, x)
            model.invoke()
            return model.get_tensor(output_idx)[None, ]
        
        for name, path in self.model_paths.items():
            model = tflite.Interpreter(path, num_threads = 1)
            model.allocate_tensors()
            input_idx = model.get_input_details()[0]['index']
            output_idx = model.get_output_details()[0]['index']
            self.models[name] = model
            self.predict_fns = ft.partial(predict, model, input_idx, output_idx)
            self.model_inputs[name] = model.get_input_details()[0]['shape'][1]


    ### METHODS ###
    def get_positive_prediction_frames(
                self,
                path: str,
                threshold: float = 0.5,
                return_type: Literal['audio', 'features'] = 'features',
                model_patiences: dict[str, int] = {},
                model_thresholds: dict[str, float] = {},
                debounce_time: float = 0.0
            ) -> dict[str, npt.NDArray[np.float32]]:
        '''
        Gets predictions for the input audio data, and returns the audio
        features (embeddings) or audio data for all of the frames with a score
        above the `threshold` argument. Can be a useful way to collect
        false-positive predictions.

        Args:
            path (`str`):
                The path to a 16-bit 16khz WAV audio file to process
            threshold (`float`):
                The minimum score required for a frame of audio features to be
                returned.
            return_type (`Literal['audio', 'features']`):
                The type of data to return when a positive prediction is
                detected. Can be either 'features' or 'audio' to return audio
                embeddings or raw audio data, respectively.
            model_patiences (`dict[str, int]`):
                How many consecutive frames (of 1280 samples or 80 ms) above
                the threshold that must be observed before the current frame
                will be returned as non-zero. Must be provided as an a `dict`
                where the keys are the model names and the values are the
                number of frames. Can reduce false-positive detections at the
                cost of a lower true-positive rate. By default, this behavior
                is disabled.
            model_thresholds (`dict[str, float]`):
                The threshold values to use when the `patience` or
                `debounce_time` behavior is enabled. Must be provided as a
                `dict` where the keys are the model names and the values are
                the thresholds.
            debounce_time (`float`):
                The time (in seconds) to wait before returning another non-zero
                prediction after a non-zero prediction. Can preven multiple
                detections of the same wake-word.

        Returns:
            positive_frames (`dict[str, ndarray]`):
                A `dict` with filenames as keys and `N` x `M` arrays as values,
                where `N` is the number of examples and `M` is the number of
                audio features, depending on the model input shape.
        '''
        # Load audio clip as 16-bit PCM data
        with wave.open(path, mode='rb') as f:
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

        # Iterate through clip, getting predictions
        positive_data = defaultdict(list)
        step_size = 1280
        for i in range(0, data.shape[0]-step_size, step_size):
            predictions = self._predict(
                data[i:i+step_size],
                model_patiences,
                model_thresholds,
                debounce_time
            )
            for cls in predictions.keys():
                if predictions[cls] >= threshold:
                    if cls in Model.TIMER_MAPPINGS.values():
                        parent_model = 'timer'
                    else:
                        parent_model = cls
                    features = self.preprocessor.get_features(
                        self.model_inputs[parent_model]
                    )
                    if return_type == 'features':
                        positive_data[cls].append(features)
                    if return_type == 'audio':
                        context = data[max(0, i - 16000*3):i + 16000]
                        if len(context) == 16000*4:
                            positive_data[cls].append(context)

        positive_data_combined = {}
        for cls in positive_data.keys():
            positive_data_combined[cls] = np.vstack(positive_data[cls])

        return positive_data_combined
    
    
    def multipredict (
                self,
                x: npt.NDArray[np.int16],
                model_patiences: dict[str, int] = {},
                model_thresholds: dict[str, float] = {},
                debounce_time: float = 0.0
            ) -> dict[str, float]:
        '''
        Predict with all of the wakeword models on the input audio frames.

        Args:
            x (`ndarray`):
                The input audio data to predict on with the models. Ideally
                should be multiples of 80 ms (1280 samples), with longer
                lengths reducing overall CPU usage but decreasing detection
                latency. Input audio with durations greater than or less than
                80 ms is also supported, though this will add a detection delay
                of up to 80 ms as the appropriate number of samples are
                accumulated.
            model_patiences (`dict[str, int]`):
                How many consecutive frames (of 1280 samples or 80 ms) above
                the threshold that must be observed before the current frame
                will be returned as non-zero. Must be provided as an a `dict`
                where the keys are the model names and the values are the
                number of frames. Can reduce false-positive detections at the
                cost of a lower true-positive rate. By default, this behavior
                is disabled.
            model_thresholds (`dict[str, float]`):
                The threshold values to use when the `patience` or
                `debounce_time` behavior is enabled. Must be provided as a
                `dict` where the keys are the model names and the values are
                the thresholds.
            debounce_time (`float`):
                The time (in seconds) to wait before returning another non-zero
                prediction after a non-zero prediction. Can preven multiple
                detections of the same wake-word.

        Returns:
            results (`dict[str, float]`):
                A `dict` mapping model names to prediction scores between 0 and
                1.
        '''
        return self._predict(
            x,
            model_patiences,
            model_thresholds,
            debounce_time
        )


    def multipredict_clip (
                self,
                clip: Union[str, np.ndarray],
                padding: int = 1,
                chunk_size = 1280,
                model_patiences: dict[str, int] = {},
                model_thresholds: dict[str, float] = {},
                debounce_time: float = 0.0
            ) -> list[dict[str, float]]:
        '''
        Predict on an full audio clip, simulating streaming prediction. The
        input clip must bit a 16-bit, 16 khz, single-channel WAV file.

        Args:
            clip (`Union[str, ndarray]`):
                The path to a 16-bit PCM, 16 khz, single-channel WAV file, or
                a 1D `ndarray` containing the same type of data.
            padding (`int`):
                How many seconds of silence to pad the start/end of the clip
                with to make sure that short clips can be processed correctly
                (defaults to 1).
            chunk_size (`int`):
                The size (in samples) of each chunk of audio to pass to the
                model.
            model_patiences (`dict[str, int]`):
                How many consecutive frames (of 1280 samples or 80 ms) above
                the threshold that must be observed before the current frame
                will be returned as non-zero. Must be provided as an a `dict`
                where the keys are the model names and the values are the
                number of frames. Can reduce false-positive detections at the
                cost of a lower true-positive rate. By default, this behavior
                is disabled.
            model_thresholds (`dict[str, float]`):
                The threshold values to use when the `patience` or
                `debounce_time` behavior is enabled. Must be provided as a
                `dict` where the keys are the model names and the values are
                the thresholds.
            debounce_time (`float`):
                The time (in seconds) to wait before returning another non-zero
                prediction after a non-zero prediction. Can preven multiple
                detections of the same wake-word.

        Returns:
            results (`list[dict[str, float]]`):
                A `list` containing the frame-level predictions for the audio
                clip.
        '''
        if isinstance(clip, str):
            # Load audio clip as 16-bit PCM data
            with wave.open(clip, mode='rb') as f:
                data = np.frombuffer(
                    f.readframes(f.getnframes()), dtype=np.int16
                )
        elif isinstance(clip, np.ndarray):
            data = clip

        if padding:
            data = np.concatenate(
                (
                    np.zeros(16000*padding).astype(np.int16),
                    data,
                    np.zeros(16000*padding).astype(np.int16)
                )
            )

        # Iterate through clip, getting predictions
        predictions: list[dict[str, float]] = []
        step_size = chunk_size
        for i in range(0, data.shape[0]-step_size, step_size):
            predictions.append(self._predict(
                data[i:i+step_size],
                model_patiences,
                model_thresholds,
                debounce_time
            ))

        return predictions
    
    
    def multipredict_with_timings (
                self,
                x: npt.NDArray[np.int16],
                model_patiences: dict[str, int] = {},
                model_thresholds: dict[str, float] = {},
                debounce_time: float = 0.0
            ) -> tuple[dict[str, float], dict[str, float]]:
        '''
        Predict with all of the wakeword models on the input audio frames and
        gather timing information.

        Args:
            x (`ndarray`):
                The input audio data to predict on with the models. Ideally
                should be multiples of 80 ms (1280 samples), with longer
                lengths reducing overall CPU usage but decreasing detection
                latency. Input audio with durations greater than or less than
                80 ms is also supported, though this will add a detection delay
                of up to 80 ms as the appropriate number of samples are
                accumulated.
            model_patiences (`dict[str, int]`):
                How many consecutive frames (of 1280 samples or 80 ms) above
                the threshold that must be observed before the current frame
                will be returned as non-zero. Must be provided as an a `dict`
                where the keys are the model names and the values are the
                number of frames. Can reduce false-positive detections at the
                cost of a lower true-positive rate. By default, this behavior
                is disabled.
            model_thresholds (`dict[str, float]`):
                The threshold values to use when the `patience` or
                `debounce_time` behavior is enabled. Must be provided as a
                `dict` where the keys are the model names and the values are
                the thresholds.
            debounce_time (`float`):
                The time (in seconds) to wait before returning another non-zero
                prediction after a non-zero prediction. Can preven multiple
                detections of the same wake-word.

        Returns:
            results (`tuple[dict[str, float], dict[str, float]]`):
                A `tuple` of two `dicts`; the first mapping model names to
                prediction scores and the second mapping model names,
                'preprocessing', and 'vad' keys to timing information.
        '''
        return self._predict(
            x,
            model_patiences,
            model_thresholds,
            debounce_time,
            True
        )


    def predict (
                self,
                x: npt.NDArray[np.float16],
                patience: Optional[int] = None,
                threshold: Optional[float] = None,
                debounce_time: float = 0.0
            ) -> float:
        '''
        Predict with a single wakeword model on the input audio frames.

        Args:
            x (`ndarray`):
                The input audio data to predict on with the model. Ideally
                should be multiples of 80 ms (1280 samples), with longer
                lengths reducing overall CPU usage but decreasing detection
                latency. Input audio with durations greater than or less than
                80 ms is also supported, though this will add a detection delay
                of up to 80 ms as the appropriate number of samples are
                accumulated.
            patience (`Optional[int]`):
                How many consecutive frames (of 1280 samples or 80 ms) above
                the threshold that must be observed before the current frame
                will be returned as non-zero. Can reduce false-positive
                detections at the cost of a lower true-positive rate. By
                default, this behavior is disabled.
            threshold (`Optional[float]`):
                The threshold value to use when the `patience` or
                `debounce_time` behavior is enabled.
            debounce_time (`float`):
                The time (in seconds) to wait before returning another non-zero
                prediction after a non-zero prediction. Can preven multiple
                detections of the same wake-word.

        Returns:
            result (`float`):
                A prediction score between 0 and 1.
        '''
        if not self.is_singleton:
            raise WrongPredictMethodError()
        
        patiences = {} if patience is None else { self.singleton: patience }
        thresholds = {} if threshold is None else { self.singleton: threshold }

        results = self._predict(x, patiences, thresholds, debounce_time)

        return results[self.singleton]


    def predict_clip (
                self,
                clip: Union[str, np.ndarray],
                padding: int = 1,
                chunk_size = 1280,
                patience: Optional[int] = None,
                threshold: Optional[float] = None,
                debounce_time: float = 0.0
            ) -> list[dict[str, float]]:
        '''
        Predict on an full audio clip, simulating streaming prediction. The
        input clip must bit a 16-bit, 16 khz, single-channel WAV file.

        Args:
            clip (`Union[str, ndarray]`):
                The path to a 16-bit PCM, 16 khz, single-channel WAV file, or
                a 1D `ndarray` containing the same type of data.
            padding (`int`):
                How many seconds of silence to pad the start/end of the clip
                with to make sure that short clips can be processed correctly
                (defaults to 1).
            chunk_size (`int`):
                The size (in samples) of each chunk of audio to pass to the
                model.
            patience (`Optional[int]`):
                How many consecutive frames (of 1280 samples or 80 ms) above
                the threshold that must be observed before the current frame
                will be returned as non-zero. Can reduce false-positive
                detections at the cost of a lower true-positive rate. By
                default, this behavior is disabled.
            threshold (`Optional[float]`):
                The threshold value to use when the `patience` or
                `debounce_time` behavior is enabled.
            debounce_time (`float`):
                The time (in seconds) to wait before returning another non-zero
                prediction after a non-zero prediction. Can preven multiple
                detections of the same wake-word.

        Returns:
            results (`list[float]`):
                A `list` containing the frame-level predictions for the audio
                clip.
        '''
        if not self.is_singleton:
            raise WrongPredictMethodError()
        
        patiences = {} if patience is None else { self.singleton: patience }
        thresholds = {} if threshold is None else { self.singleton: threshold }

        results = self.multipredict_clip(
            clip,
            padding,
            chunk_size,
            patiences,
            thresholds,
            debounce_time
        )

        return [res[self.singleton] for res in results]
    

    def predict_with_timings (
                self,
                x: npt.NDArray[np.float16],
                patience: Optional[int] = None,
                threshold: Optional[float] = None,
                debounce_time: float = 0.0
            ) -> tuple[float, dict[str, float]]:
        '''
        Predict with a single wakeword model on the input audio frames and
        gather timing information.

        Args:
            x (`ndarray`):
                The input audio data to predict on with the model. Ideally
                should be multiples of 80 ms (1280 samples), with longer
                lengths reducing overall CPU usage but decreasing detection
                latency. Input audio with durations greater than or less than
                80 ms is also supported, though this will add a detection delay
                of up to 80 ms as the appropriate number of samples are
                accumulated.
            patience (`Optional[int]`):
                How many consecutive frames (of 1280 samples or 80 ms) above
                the threshold that must be observed before the current frame
                will be returned as non-zero. Can reduce false-positive
                detections at the cost of a lower true-positive rate. By
                default, this behavior is disabled.
            threshold (`Optional[float]`):
                The threshold value to use when the `patience` or
                `debounce_time` behavior is enabled.
            debounce_time (`float`):
                The time (in seconds) to wait before returning another non-zero
                prediction after a non-zero prediction. Can preven multiple
                detections of the same wake-word.

        Returns:
            result (`tuple[float, dict[str, float]]`):
                A `tuple` containing a `float` prediction score and a timing
                `dict` mapping 'preprocessor', 'vad', and the model name as
                keys to `float` time values.
        '''
        if not self.is_singleton:
            raise WrongPredictMethodError()
        
        patiences = {} if patience is None else { self.singleton: patience }
        thresholds = {} if threshold is None else { self.singleton: threshold }

        scores, timings = self._predict(
            x,
            patiences,
            thresholds,
            debounce_time,
            True
        )

        return scores[self.singleton], timings
    

    def reset(self):
        '''
        Reset the prediction and audio feature buffers. Useful for
        re-initializing the model, though may not be efficient when called too
        frequently.
        '''
        self.prediction_buffer = defaultdict(ft.partial(deque, maxlen=30))
        self.preprocessor.reset()
