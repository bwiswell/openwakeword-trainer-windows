from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

from ..resources import Resource


@dataclass
class DataSpec:

    _final_path: Path
    _download_path: Optional[Path] = None

    _ensures: Optional[list[Path]] = None
    _recreates: Optional[list[Path]] = None

    _resources: Optional[list[Resource]] = None

    def __init__ (
                self,
                final_path: Path,
                download_path: Optional[Path] = None,
                ensures: Optional[list[Path]] = None,
                recreates: Optional[list[Path]] = None,
                resources: Optional[list[Resource]] = None
            ):
        self._final_path = final_path
        self._download_path = download_path
        self._ensures = ensures
        self._recreates = recreates
        self._resources = resources


    ### METHODS ###
    def download (self):
        if self._resources is not None:
            for resource in self._resources:
                resource.download(self._download_path)


    def ensure (self):
        if self._ensures is not None:
            for path in self._ensures:
                path.mkdir(parents=True, exist_ok=True)
        if self._recreates is not None:
            for path in self._recreates:
                if path.exists(): shutil.rmtree(path)
                path.mkdir(parents=True)


    def unpack (self):
        if self._resources is not None:
            for resource in self._resources:
                resource.unpack(self._download_path, self._final_path)