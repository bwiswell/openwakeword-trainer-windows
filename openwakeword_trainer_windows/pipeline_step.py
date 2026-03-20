from enum import Enum


class PipelineStep(Enum):
    ENSURE = 0
    DOWNLOAD = 1
    UNPACK = 2
    PATCH = 3
    CONFIGURE = 4
    RECORD = 5
    TTS = 6
    AUGMENT = 7
    TRAIN = 8
    EXPORT = 9