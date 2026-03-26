from pathlib import Path

from .common import InferenceFramework, ModelNotFoundError
from .resource import Resource


class Downloader:

    MODELS = {
        'alexa': Resource('alexa_v0.1'),
        'hey_jarvis': Resource('hey_jarvis_v0.1'),
        'hey_mycroft': Resource('hey_mycroft_v0.1'),
        'hey_rhasspy': Resource('hey_rhasspy_v0.1'),
        'timer': Resource('timer_v0.1'),
        'weather': Resource('weather_v0.1')
    }

    PREPROCESSORS = [
        Resource('embedding_model'),
        Resource('melspectrogram')
    ]

    VAD = Resource(
        name = 'silero_vad',
        only = 'onnx'
    )


    def __init__ (
                self,
                model_path: Path,
                inference_framework: InferenceFramework
            ):
        self.framework = inference_framework
        self.model_path = model_path


    ### METHODS ###
    def ensure_preprocessors (self):
        for preprocessor in Downloader.PREPROCESSORS:
            _ = preprocessor.download(self.model_path, self.framework)


    def ensure_models (self, models: list[str]) -> list[Path]:
        paths: list[Path] = []
        for model in models:
            if model not in Downloader.MODELS:
                raise ModelNotFoundError(model, False, False)
            paths.append(
                Downloader.MODELS[model].download(
                    self.model_path,
                    self.framework
                )
            )
        return paths


    def ensure_vad (self):
        _ = Downloader.VAD.download(self.model_path, self.framework)