from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess

from ..logger import Logger

from .resource import Resource


@dataclass
class GitResource(Resource):

    data_dirs: list[list[str]] = field(default_factory = lambda: [])

    ### METHODS ###
    def download (self, local_path: Path):
        if self.is_downloaded(local_path):
            Logger.log(f'✅ {self.name} is already downloaded')
        else:
            dest = self.path(local_path)
            if dest.exists(): shutil.rmtree(dest)
            Logger.log(f'🔄 downloading {self.name}...')
            try:
                subprocess.run(['git', 'clone', self.remote, str(dest)])
                Logger.log(f'✅ {self.name} successfully downloaded')
            except:
                Logger.log(f'❌ failed to download {self.name}')
                raise Exception()

    def is_downloaded (self, local_path: Path) -> bool:
        path = self.path(local_path)
        if not path.exists(): return False
        return len(list(path.iterdir())) > 1
    
    def is_unpacked (self, local_path: Path) -> bool:
        path = self.path(local_path)
        for data_dir_slugs in self.data_dirs:
            dir_path = path
            for slug in data_dir_slugs:
                dir_path /= slug
            if not dir_path.exists():
                return False
        return True
    
    def unpack (self, data_path: Path, dest_path: Path):
        if self.is_unpacked(dest_path):
            Logger.log(f'✅ {self.name} is already unpacked')
        else:
            Logger.log(f'📦 unpacking {self.name}...')
            path = self.path(dest_path)
            for data_dir_slugs in self.data_dirs:
                dir_path = path
                for slug in data_dir_slugs:
                    dir_path /= slug
                dir_path.mkdir(parents=True, exist_ok=True)
            Logger.log(f'✅ {self.name} successfully unpacked')


OPENWAKEWORD = GitResource(
    name = 'openwakeword',
    remote = 'https://github.com/dscripka/openWakeWord',
    data_dirs = [['openwakeword', 'resources', 'models']]
)

GIT_RESOURCES = [OPENWAKEWORD]