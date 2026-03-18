from dataclasses import dataclass
from pathlib import Path
from urllib import request

from ..logger import Logger

from .resource import Resource


@dataclass
class FeatureResource(Resource):

    ### METHODS ###
    def download (self, local_path: Path):
        if self.is_downloaded(local_path):
            Logger.log(f'✅ {self.name} is already downloaded')
        else:
            dest = self.path(local_path)
            Logger.log(f'🔄 downloading {self.name}...')
            try:
                url = self.remote + self.name
                request.urlretrieve(self.remote + self.name, str(dest))
                Logger.log(f'✅ {self.name} successfully downloaded')
            except:
                Logger.log(f'❌ failed to download {self.name}')
                raise Exception()

    def is_downloaded (self, local_path: Path) -> bool:
        return self.path(local_path).exists()
    
    def is_unpacked (self, local_path: Path) -> bool:
        return True
    
    def path (self, local_path: Path) -> Path:
        return local_path / self.name
    
    def unpack (self, data_path: Path, dest_path: Path):
        Logger.log(f'✅ {self.name} is already unpacked')


OWW_FEATURES = FeatureResource(
    name = 'openwakeword_features_ACAV100M_2000_hrs_16bit.npy',
    remote = 'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/'
)

VALIDATION_FEATURES = FeatureResource(
    name = 'validation_set_features.npy',
    remote = 'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/'
)

FEATURE_RESOURCES = [OWW_FEATURES, VALIDATION_FEATURES]