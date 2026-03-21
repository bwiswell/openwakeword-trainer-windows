from dataclasses import dataclass
import io
from pathlib import Path
import shutil
import time

import datasets as ds
import huggingface_hub as hf
import soundfile as sf

from ...logger import Logger

from .resource import Resource


@dataclass
class WavResource(Resource):

    parquets: int
    wavs: int

    ### METHODS ###
    def download (self, local_path: Path):
        if self.is_downloaded(local_path):
            Logger.log(f'✅ {self.name} is already downloaded')
        else:
            Logger.log(f'🔄 downloading {self.name}...')
            hf.snapshot_download(
                repo_id = self.remote,
                cache_dir = str(local_path / 'cache'),
                local_dir = str(local_path / self.name),
                repo_type = 'dataset',
                max_workers = 4
            )
            time.sleep(2)
            Logger.log(f'\n✅ {self.name} successfully downloaded')

    def is_downloaded (self, local_path: Path) -> bool:
        path = local_path / self.name
        if not path.exists():
            return False
        data_path = path / 'data'
        count = len(list(data_path.glob('*.parquet')))
        if self.parquets != count:
            return False
        return True
    
    def is_unpacked (self, local_path: Path) -> bool:
        path = local_path / self.name
        if not path.exists():
            return False
        count = len(list(path.glob('*.wav')))
        if self.wavs != count:
            return False
        return True
    
    def unpack (self, data_path: Path, dest_path: Path):
        if self.is_unpacked(dest_path):
            Logger.log(f'✅ {self.name} is already unpacked')
        else:
            Logger.log(f'📦 unpacking {self.name}...')
            src = self.path(data_path)
            dest = self.path(dest_path)
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            data = ds.load_dataset(
                str(src),
                cache_dir = str(data_path / 'cache')
            ).cast_column('audio', ds.Audio(decode=False))
            for i, example in enumerate(data['train']):
                audio_data = example['audio']
                wav, samplerate = sf.read(io.BytesIO(audio_data['bytes']))
                filename = dest / f'sample_{i}.wav'
                sf.write(filename, wav, samplerate)
                if (i + 1) % 500 == 0:
                    print(f'\t...extracted {i + 1} files')
            Logger.log(f'✅ {self.name} successfully unpacked')


AUDIOSET = WavResource(
    name = 'audioset_16k',
    remote = 'benjamin-paine/freesound-laion-640k-commercial-16khz-tiny',
    parquets = 4,
    wavs = 20000
)

FMA = WavResource(
    name = 'fma',
    remote = 'benjamin-paine/free-music-archive-commercial-16khz-full',
    parquets = 13,
    wavs = 8802
)

MIT_RIRS = WavResource(
    name = 'mit_rirs',
    remote = 'benjamin-paine/mit-impulse-response-survey-16khz',
    parquets = 1,
    wavs = 270
)