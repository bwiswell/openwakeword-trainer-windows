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


import torch as to


class FCNBlock(to.nn.Module):

    def __init__(self, layer_dim: int):
        super().__init__()
        self.fcn_layer = to.nn.Linear(layer_dim, layer_dim)
        self.relu = to.nn.ReLU()
        self.layer_norm = to.nn.LayerNorm(layer_dim)

    def forward (self, x: to.Tensor):
        return self.relu(self.layer_norm(self.fcn_layer(x)))


class Network(to.nn.Module):

    def __init__(self, input_shape: tuple[int, int], layer_dim: int):
        super().__init__()
        self.layers = [
            to.nn.Flatten(),
            to.nn.Linear(input_shape[0] * input_shape[1], layer_dim),
            to.nn.LayerNorm(layer_dim),
            to.nn.ReLU(),
            FCNBlock(layer_dim),
            to.nn.Linear(layer_dim, 1),
            to.nn.Sigmoid()
        ]

    def forward(self, x: to.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x