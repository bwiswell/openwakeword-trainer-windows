from dataclasses import dataclass
from pathlib import Path


@dataclass
class Resource:

    name: str
    remote: str


    ### METHODS ###
    def download (self, local_path: Path):
        raise NotImplementedError

    def is_downloaded (self, local_path: Path) -> bool:
        raise NotImplementedError
    
    def is_unpacked (self, local_path: Path) -> bool:
        raise NotImplementedError
    
    def path (self, local_path: Path) -> Path:
        return local_path / self.name
    
    def unpack (self, data_path: Path, dest_path: Path):
        raise NotImplementedError