from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Optional

import requests

from .common import InferenceFramework, DownloadError, IncompatibleModelError


@dataclass
class Resource:

    BASE = 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1'

    name: str
    only: Optional[InferenceFramework] = None


    ### METHODS ###
    def download (
                self,
                local_path: Path,
                inference_framework: InferenceFramework
            ) -> Path:
        if self.only and self.only != inference_framework:
            raise IncompatibleModelError(self.name, inference_framework)
        full_name = f'{self.name}.{inference_framework}'
        path = local_path / full_name
        if path.exists():
            logging.info(f'{full_name} is already downloaded')
            return
        remote = f'{Resource.BASE}/{full_name}'
        logging.info(f'downloading {full_name}')
        try:
            with requests.get(remote, stream=True) as r:
                r.raise_for_status()
                with open(str(path), 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            logging.info(f'{full_name} successfully downloaded')
        except:
            raise DownloadError(full_name)
        return path
        

    def path (
                self,
                local_path: Path,
                inference_framework: InferenceFramework
            ) -> Path:
        full_name = f'{self.name}.{inference_framework}'
        return local_path / full_name