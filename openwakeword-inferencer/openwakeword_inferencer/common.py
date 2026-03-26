from typing import Literal


InferenceFramework = Literal['onnx', 'tflite']


class DownloadError(Exception):
    '''
    Exception raised when an automatic download was attempted, but failed.
    '''

    def __init__ (self, model_name: str):
        Exception.__init__(
            self,
            f'there was an error while downloading {model_name}.'
        )


class IncompatibleModelError(Exception):
    '''
    Exception raised when a user-specified model (or the VAD model) is used
    with an incompatible inference framework.
    '''

    def __init__ (self, model_name: str, framework: InferenceFramework):
        msg = f"{model_name} is not compatible with the '{framework}' \
                inference framework."
        Exception.__init__(self, msg)


class InferenceFrameworkError(Exception):
    '''
    Exception raised when inference framework dependencies are missing or the
    specified inference framework is not one of 'onnx' or 'tflite'.
    '''

    def __init__ (self, framework: InferenceFramework | str):
        if framework == 'onnx' or framework == 'tflite':
            msg = f"The necessary dependencies for the '{framework}' \
                    inference framework are not available. Make sure that the \
                    package was installed with the appropriate extras and the \
                    correct inference framework parameter was passed."
        else:
            msg = f"The inference framework parameter value '{framework}' is \
                    not a valid inference framework selection. Please choose \
                    either 'onnx' or 'tflite' and try again (make sure the \
                    package was installed with the appropriate extras)."
        Exception.__init__(self, msg)


class ModelNameConflictError(Exception):
    '''
    Exception raise when a custom model name conflicts with a pretrained model
    name.
    '''

    def __init__ (self, name: str):
        msg = f"The custom model {name} has the same name as a pretrained \
                model, which can cause class mapping conflicts. Please choose \
                a different name for your pretrained model."
        Exception.__init__(self, msg)


class ModelNotFoundError(Exception):
    '''
    Exception raised when a model path that does not exist or a pretrained
    model name that does not exist is passed to `Model`.
    '''

    def __init__ (self, name_or_path: str, custom: bool, verifier: bool):
        if verifier:
            msg = f"There is no verifier model located at {name_or_path}. \
                    Ensure you have specified the correct location and try \
                    again."
        elif custom:
            msg = f"There is no valid openWakeWord model located at \
                    {name_or_path}. Ensure you have specified the correct \
                    location and try again."
        else:
            msg = f"There is no pretrained model associated with the name \
                    {name_or_path}. Ensure you are using one of the valid \
                    pretrained model names: 'alexa', 'hey_jarvis', \
                    'hey_mycroft', 'hey_rhasspy', 'timer', or 'weather'."
        Exception.__init__(self, msg)


class UnknownVerifierModelError(Exception):
    '''
    Exception raised when a custom verifier model is passed but can not be
    associated with any loaded model.
    '''

    def __init__ (self, name: str):
        msg = f"No custom or pretrained model could be associated with the \
                custom verifier model for {name}."
        Exception.__init__(self, msg)


class WrongPredictMethodError(Exception):
    '''
    Exception raised when the `predict`, `predict_clip`, or
    `predict_with_timings` functions are called when multiple models are
    loaded.
    '''

    def __init__ (self):
        msg = f"Multiple models are loaded, but `predict`, `predict_clip` or \
                `predict_with_timings` was called. Use `multipredict`, \
                `multipredict_clip`, or `multipredict_with_timings` instead."
        Exception.__init__(self, msg)