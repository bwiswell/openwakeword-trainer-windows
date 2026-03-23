from typing import Literal


InferenceFramework = Literal['onnx', 'tflite']


class InferenceFrameworkError(Exception):
    '''
    Exception raised when inference framework dependencies are missing or the
    specified inference framework is not one of 'onnx' or 'tflite'.
    '''

    def __init__ (self, framework: str):
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
