# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################

# NOTICE: This file has been substantially modified by Benj Wiswell.


from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import torch as to

from ..config import Config
from ..data_manager import DataManager


TransformFn = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]


class MMapDataSplit:

    def __init__ (
                self,
                path: str,
                n: int,
                label: int,
                transform: Optional[TransformFn] = None
            ):
        self.counter = 0
        self.data: npt.NDArray[np.float32] = np.load(path, mmap_mode='r')
        self.label = label
        self.n = n
        self.shape = self.data.shape
        self.transform = transform



class MMapDataset(to.utils.data.IterableDataset):

    def __init__ (self, config: Config, dm: DataManager, n: int):
        self.n = n
        self.splits = [
            MMapDataSplit(
                dm.features.pos_train,
                config.positive_per_batch
            ),
            MMapDataSplit(
                dm.features.neg_train,
                config.negative_per_batch
            ),
            MMapDataSplit(
                dm.features.acav,
                config.acav_per_batch,
                self._transform
            )
        ]

    
    ### MAGIC METHODS ###
    def __iter__ (self):
        return self
    

    def __next__ (self):
        # Build batch
        while True:
            x: list[npt.NDArray[np.float32]] = []
            y: list[list[int]] = []
            for split in self.splits:
                # Restart at zeroth index if an array reaches the end
                if split.counter >= split.shape[0]:
                    split.counter = 0

                # Get data from mmaped file
                x_batch = split.data[split.counter:split.counter + split.n]
                split.counter += x_batch.shape[0]

                # Transform data
                if split.transform is not None:
                    x_batch = split.transform(x_batch)

                # Make labels for data
                y_batch = [split.label] * x_batch.shape[0]

                # Add data to batch
                x.append(x_batch)
                y.append(y_batch)

            return np.vstack(x), np.array(y)
        

    ### HELPERS ###
    def _transform (
                self,
                x: npt.NDArray[np.float32]
            ) -> npt.NDArray[np.float32]:
        if self.n < x.shape[1] or x.shape[1] < self.n:
            x = np.vstack(x)
            x = np.array([
                x[i:i + self.n, :]
                for i in range(0, x.shape[0] - self.n, self.n)
            ])
        return x