from enum import Enum


class PipelineStep(Enum):
    ENSURE = 0
    DOWNLOAD = 1
    UNPACK = 2
    RECORD = 3
    TTS = 4
    AUGMENT = 5
    TRAIN = 6