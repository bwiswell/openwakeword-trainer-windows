import shutil
import time
import wave

import pyaudio as pa

from .config import Config
from .data_manager import DataManager
from .logger import Logger


class Recorder:

    CHANNELS = 1
    CHUNK = 1024
    DURATION = 2.0
    FORMAT = pa.paInt16
    SAMPLE_RATE = 16000

    def __init__ (self, config: Config, dm: DataManager):
        self.config = config
        self.dm = dm


    ### HELPERS ###
    def _record (self, path: str):
        print('recording in: 2...', end='\r')
        time.sleep(1)
        print('recording in: 1...', end='\r')
        time.sleep(1)
        print('recording - say your phrase now')

        audio = pa.PyAudio()
        stream = audio.open(
            rate = Recorder.SAMPLE_RATE,
            channels = Recorder.CHANNELS,
            format = Recorder.FORMAT,
            input = True,
            frames_per_buffer = Recorder.SAMPLE_RATE
        )

        chunks = int(Recorder.SAMPLE_RATE / Recorder.CHUNK * Recorder.DURATION)
        frames = [stream.read(Recorder.CHUNK) for _ in range(chunks)]

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(path, 'wb') as wf:
            wf.setnchannels(Recorder.CHANNELS)
            wf.setsampwidth(audio.get_sample_size(Recorder.FORMAT))
            wf.setframerate(Recorder.SAMPLE_RATE)
            wf.writeframes(b''.join(frames))


    ### METHODS ###
    def record_samples (self):
        Logger.log('🚀 recording custom samples...')

        names = ('positive', 'negative')
        existing_records = (self.dm.n_recorded_pos, self.dm.n_recorded_neg)
        record_paths = (self.dm.record_pos_path, self.dm.record_neg_path)
        all_phrases = (self.config.target_phrases, self.config.negative_phrases)

        for name, idx, record_path, phrases in zip(
                    names,
                    existing_records,
                    record_paths,
                    all_phrases
                ):
            Logger.log(f'🔄 recording {name} samples...')
            i = 0
            for phrase in phrases:
                while True:
                    cmd = input(
                        f"[{phrase}] press ENTER to record or 'q' to quit: "
                    )
                    if cmd.lower() == 'q': break
                    self._record(str(record_path / f'sample_{idx + i}.wav'))
                    i += 1

        Logger.log('✨ custom validator samples recorded')